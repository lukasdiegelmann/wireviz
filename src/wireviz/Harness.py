# -*- coding: utf-8 -*-

import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, List, Union

from graphviz import Graph, Source
from wireviz import APP_NAME, APP_URL, __version__, wv_colors
from wireviz.DataClasses import (
    Cable,
    Connector,
    MateComponent,
    MatePin,
    Metadata,
    Options,
    Side,
    Tweak,
)
from wireviz.svgembed import embed_svg_images, embed_svg_images_file
from wireviz.wv_bom import (
    HEADER_MPN,
    HEADER_PN,
    HEADER_SPN,
    bom_list,
    component_table_entry,
    generate_bom,
    get_additional_component_table,
    pn_info_string,
)
from wireviz.wv_colors import get_color_hex, translate_color
from wireviz.wv_gv_html import (
    html_bgcolor,
    html_bgcolor_attr,
    html_caption,
    html_colorbar,
    html_image,
    html_line_breaks,
    nested_html_table,
    remove_links,
)
from wireviz.wv_helper import (
    awg_equiv,
    file_write_text,
    flatten2d,
    is_arrow,
    mm2_equiv,
    tuplelist2tsv,
)
from wireviz.wv_html import generate_html_output

OLD_CONNECTOR_ATTR = {
    "pinout": "was renamed to 'pinlabels' in v0.2",
    "pinnumbers": "was renamed to 'pins' in v0.2",
    "autogenerate": "is replaced with new syntax in v0.4",
}


def check_old(node: str, old_attr: dict, args: dict) -> None:
    """Raise exception for any outdated attributes in args."""
    for attr, descr in old_attr.items():
        if attr in args:
            raise ValueError(f"'{attr}' in {node}: '{attr}' {descr}")


def smooth_edges(layout: str, width: float = 2.0) -> str:
    """
    Redraw the edges of a laid-out dot source (`-Tdot` output) as smooth curves
    with horizontal tangents at both ends, like wires leaving connector pins.

    Each edge becomes one cubic Bezier S-curve from its start to its end point.
    If that curve would cross a node, the route found by dot is kept, but
    smoothed: a Catmull-Rom curve through its on-curve points.
    """
    # node bounding boxes (points), from height/pos/width in inches
    boxes = []
    for m in re.finditer(
        r'height=([\d.]+),.*?pos="([\d.-]+),([\d.-]+)",.*?width=([\d.]+)',
        layout,
        re.S,
    ):
        h, x, y, w = (float(v) for v in m.groups())
        h, w = h * 72, w * 72
        boxes.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))

    def inside(px, py, box, margin=0.0):
        x0, y0, x1, y1 = box
        return x0 - margin < px < x1 + margin and y0 - margin < py < y1 + margin

    def bezier(p0, p1, p2, p3, n=40):
        for i in range(n + 1):
            t = i / n
            u = 1 - t
            yield (
                u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
            )

    CLEARANCE = 14  # minimum distance between a wire and other nodes (points)

    def obstacles(start, end):
        # all nodes except the ones the edge starts and ends at
        return [b for b in boxes if not (inside(*start, b, 2) or inside(*end, b, 2))]

    def collides(segments, others, margin=CLEARANCE):
        return any(
            inside(px, py, b, margin)
            for seg in segments
            for px, py in bezier(*seg)
            for b in others
        )

    def line_clear(p, q, others, margin):
        n = max(int(math.hypot(q[0] - p[0], q[1] - p[1]) / 4), 1)
        return not any(
            inside(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n, b, margin)
            for i in range(n + 1)
            for b in others
        )

    def plan(start, end, others):
        """
        Shortest left-to-right path from start to end through the corners of
        the obstacles, inflated so the path keeps its distance from them.
        Returns the waypoints (without start and end), or None.
        """
        sign = 1 if end[0] >= start[0] else -1
        lead_out = (start[0] + sign * 15, start[1])  # leave the pin horizontally
        lead_in = (end[0] - sign * 15, end[1])
        pad = CLEARANCE + 6
        corners = [
            (x, y)
            for x0, y0, x1, y1 in others
            for x in (x0 - pad, x1 + pad)
            for y in (y0 - pad, y1 + pad)
            if sign * (x - lead_out[0]) > 0 and sign * (lead_in[0] - x) > 0
            and not any(inside(x, y, b, CLEARANCE) for b in others)
        ]
        nodes = sorted([lead_out] + corners, key=lambda p: sign * p[0]) + [lead_in]
        cost = {0: 0.0}
        prev = {}
        for j in range(1, len(nodes)):
            for i in range(j):
                if i not in cost or sign * (nodes[j][0] - nodes[i][0]) <= 1:
                    continue
                c = cost[i] + math.hypot(
                    nodes[j][0] - nodes[i][0], nodes[j][1] - nodes[i][1]
                )
                if c < cost.get(j, math.inf) and line_clear(
                    nodes[i], nodes[j], others, CLEARANCE
                ):
                    cost[j], prev[j] = c, i
        last = len(nodes) - 1
        if last not in cost:
            return None
        path, k = [], last
        while k != 0:
            path.append(nodes[k])
            k = prev[k]
        return [nodes[0]] + path[::-1]

    def s_curve(p0, p3):
        d = max(abs(p3[0] - p0[0]) / 2, 20)
        sign = 1 if p3[0] >= p0[0] else -1  # edges normally run left to right
        return [(p0, (p0[0] + sign * d, p0[1]), (p3[0] - sign * d, p3[1]), p3)]

    def through(points, soft=False):
        # Curve through points that is monotone in x (no overshoot or hooks):
        # Hermite segments with Fritsch-Butland slopes, horizontal at the ends.
        n = len(points)
        secants = [
            (y1 - y0) / (x1 - x0)
            for (x0, y0), (x1, y1) in zip(points, points[1:])
        ]
        slopes = [0.0] * n
        for i in range(1, n - 1):
            d0, d1 = secants[i - 1], secants[i]
            if soft:  # Catmull-Rom like: smoother, but may overshoot a little
                slopes[i] = (d0 + d1) / 2
            elif d0 * d1 > 0:
                slopes[i] = 2 / (1 / d0 + 1 / d1)
        segments = []
        for i in range(n - 1):
            (x0, y0), (x3, y3) = points[i], points[i + 1]
            h = (x3 - x0) / 3
            segments.append(
                ((x0, y0), (x0 + h, y0 + slopes[i] * h), (x3 - h, y3 - slopes[i + 1] * h), (x3, y3))
            )
        return segments

    def smooth(match: re.Match) -> str:
        tokens = match[1].replace("\\\n", "").split()
        arrows = [t for t in tokens if t.startswith(("s,", "e,"))]
        points = [tuple(map(float, t.split(","))) for t in tokens if t not in arrows]
        if len(points) < 4:
            return match[0]  # not an edge spline
        start, end = points[0], points[-1]
        others = obstacles(start, end)
        segments = s_curve(start, end)
        if collides(segments, others):
            waypoints = plan(start, end, others)
            if waypoints is None:  # no clear path, follow dot's route
                waypoints = [p for p in points[3:-1:3]]

            def route(inner):
                pts = [start] + inner + [end]
                for soft in (True, False):
                    curve = through(pts, soft=soft)
                    if not collides(curve, others):
                        return curve
                return None

            # densify until the smoothed curve keeps its distance
            for _ in range(3):
                if route(waypoints) is not None:
                    break
                pts = [start] + waypoints + [end]
                denser = []
                for p, q in zip(pts, pts[1:]):
                    denser += [p, ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)]
                waypoints = denser[1:]
            # then drop every waypoint that is not needed
            i = 0
            while i < len(waypoints):
                fewer = waypoints[:i] + waypoints[i + 1 :]
                if route(fewer) is not None:
                    waypoints = fewer
                else:
                    i += 1
            segments = route(waypoints) or through([start] + waypoints + [end])
        return arrows, segments

    def fmt(points):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    def polyline(segments):
        # sampling of the curve (about every 6 points), so stripes can be
        # offset; at most ~250 samples, as Graphviz limits strings to 16 kB
        lengths = [math.hypot(seg[3][0] - seg[0][0], seg[3][1] - seg[0][1]) for seg in segments]
        step = max(6, sum(lengths) / 250)
        pts = [segments[0][0]]
        for seg, length in zip(segments, lengths):
            n = max(int(length / step), 4)
            pts.extend(list(bezier(*seg, n=n))[1:])
        return pts

    def offset(pts, dist):
        out = []
        for i, (x, y) in enumerate(pts):
            (xa, ya), (xb, yb) = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
            dx, dy = xb - xa, yb - ya
            norm = (dx * dx + dy * dy) ** 0.5 or 1
            out.append((x - dy / norm * dist, y + dx / norm * dist))
        return out

    def as_bezier(pts):
        # smooth curve through the sampled points (Catmull-Rom as cubic Bezier)
        flat = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1 = pts[max(i - 1, 0)], pts[i]
            p2, p3 = pts[i + 1], pts[min(i + 2, len(pts) - 1)]
            flat += [
                (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6),
                (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6),
                p2,
            ]
        return flat

    def edge(match: re.Match) -> str:
        head, attrs = match[1], match[2]
        pos = re.search(r'pos="([^"]*)"', attrs)
        if not pos:
            return match[0]
        result = smooth(pos)
        if isinstance(result, str):
            return match[0]
        arrows, segments = result
        color = re.search(r'color="([^"]*)"', attrs)
        # edges with the default color carry no color attribute of their own
        colors = (color[1] if color else default_color).split(":")
        if len(colors) < 2:  # single line, e.g. dashed mate
            flat = [segments[0][0]] + [p for seg in segments for p in seg[1:]]
            new_pos = 'pos="' + " ".join(arrows + [fmt(flat)]) + '"'
            return match[0].replace(pos[0], new_pos)
        # Multi-colored wire: one edge per stripe, offset perpendicular to the
        # curve so the wire keeps a constant diameter, also in steep sections.
        pts = polyline(segments)
        stripes = []
        for i, c in enumerate(colors):
            dist = (i - (len(colors) - 1) / 2) * width
            stripe_attrs = attrs.replace(pos[0], f'pos="{fmt(as_bezier(offset(pts, dist)))}"')
            stripe_attrs = re.sub(r'color="[^"]*",?\s*', "", stripe_attrs)
            stripe_attrs = re.sub(r"style=bold,?\s*", "", stripe_attrs)
            stripe_attrs = (
                f'color="{c}", penwidth={width * 1.15:.2f}, style=solid, '
                + stripe_attrs
            )
            stripes.append(f"{head}\t[{stripe_attrs}];")
        return "\n".join(stripes)

    default = re.search(r'(?m)^\tedge \[[^\]]*?color="([^"]*)"', layout)
    default_color = default[1] if default else "#000000"
    return re.sub(r"(?ms)^(\t[^\n\[]*?--[^\n\[]*?)\s*\[(.*?)\];$", edge, layout)


@dataclass
class Harness:
    metadata: Metadata
    options: Options
    tweak: Tweak

    def __post_init__(self):
        self.connectors = {}
        self.cables = {}
        self.mates = []
        self._bom = []  # Internal Cache for generated bom
        self.additional_bom_items = []

    def add_connector(self, name: str, *args, **kwargs) -> None:
        check_old(f"Connector '{name}'", OLD_CONNECTOR_ATTR, kwargs)
        self.connectors[name] = Connector(name, *args, **kwargs)

    def add_cable(self, name: str, *args, **kwargs) -> None:
        self.cables[name] = Cable(name, *args, **kwargs)

    def add_mate_pin(self, from_name, from_pin, to_name, to_pin, arrow_type) -> None:
        self.mates.append(MatePin(from_name, from_pin, to_name, to_pin, arrow_type))
        self.connectors[from_name].activate_pin(from_pin, Side.RIGHT)
        self.connectors[to_name].activate_pin(to_pin, Side.LEFT)

    def add_mate_component(self, from_name, to_name, arrow_type) -> None:
        self.mates.append(MateComponent(from_name, to_name, arrow_type))

    def add_bom_item(self, item: dict) -> None:
        self.additional_bom_items.append(item)

    def connect(
        self,
        from_name: str,
        from_pin: (int, str),
        via_name: str,
        via_wire: (int, str),
        to_name: str,
        to_pin: (int, str),
    ) -> None:
        # check from and to connectors
        for name, pin in zip([from_name, to_name], [from_pin, to_pin]):
            if name is not None and name in self.connectors:
                connector = self.connectors[name]
                # check if provided name is ambiguous
                if pin in connector.pins and pin in connector.pinlabels:
                    if connector.pins.index(pin) != connector.pinlabels.index(pin):
                        raise Exception(
                            f"{name}:{pin} is defined both in pinlabels and pins, for different pins."
                        )
                    # TODO: Maybe issue a warning if present in both lists but referencing the same pin?
                if pin in connector.pinlabels:
                    if connector.pinlabels.count(pin) > 1:
                        raise Exception(f"{name}:{pin} is defined more than once.")
                    index = connector.pinlabels.index(pin)
                    pin = connector.pins[index]  # map pin name to pin number
                    if name == from_name:
                        from_pin = pin
                    if name == to_name:
                        to_pin = pin
                if not pin in connector.pins:
                    raise Exception(f"{name}:{pin} not found.")

        # check via cable
        if via_name in self.cables:
            cable = self.cables[via_name]
            # check if provided name is ambiguous
            if via_wire in cable.colors and via_wire in cable.wirelabels:
                if cable.colors.index(via_wire) != cable.wirelabels.index(via_wire):
                    raise Exception(
                        f"{via_name}:{via_wire} is defined both in colors and wirelabels, for different wires."
                    )
                # TODO: Maybe issue a warning if present in both lists but referencing the same wire?
            if via_wire in cable.colors:
                if cable.colors.count(via_wire) > 1:
                    raise Exception(
                        f"{via_name}:{via_wire} is used for more than one wire."
                    )
                # list index starts at 0, wire IDs start at 1
                via_wire = cable.colors.index(via_wire) + 1
            elif via_wire in cable.wirelabels:
                if cable.wirelabels.count(via_wire) > 1:
                    raise Exception(
                        f"{via_name}:{via_wire} is used for more than one wire."
                    )
                via_wire = (
                    cable.wirelabels.index(via_wire) + 1
                )  # list index starts at 0, wire IDs start at 1

        # perform the actual connection
        self.cables[via_name].connect(from_name, from_pin, via_wire, to_name, to_pin)
        if from_name in self.connectors:
            self.connectors[from_name].activate_pin(from_pin, Side.RIGHT)
        if to_name in self.connectors:
            self.connectors[to_name].activate_pin(to_pin, Side.LEFT)

    def create_graph(self) -> Graph:
        dot = Graph()
        dot.body.append(f"// Graph generated by {APP_NAME} {__version__}\n")
        dot.body.append(f"// {APP_URL}\n")
        dot.attr(
            "graph",
            rankdir="LR",
            ranksep=str(self.options.ranksep),
            bgcolor=wv_colors.translate_color(self.options.bgcolor, "HEX"),
            nodesep=str(self.options.nodesep),
            fontname=self.options.fontname,
        )  # TODO: Add graph attribute: charset="utf-8",
        dot.attr(
            "node",
            shape="none",
            width="0",
            height="0",
            margin="0",  # Actual size of the node is entirely determined by the label.
            style="filled",
            fillcolor=wv_colors.translate_color(self.options.bgcolor_node, "HEX"),
            fontname=self.options.fontname,
        )
        dot.attr("edge", style="bold", fontname=self.options.fontname)
        if self.options.wire_width != 2:
            dot.attr("edge", penwidth=str(self.options.wire_width))
        bar = max(round(self.options.wire_width), 1)  # wire bar height in cables

        # Connectors with `attach: true` are drawn inside the pin row of the
        # connector they are mated with. attachments[(host, pinindex, side)]
        # lists the attached connectors; attached_ports maps each attached
        # connector to the port (in the host) that replaces its own node.
        attachments = {}
        attached_ports = {}
        for mate in self.mates:
            if not isinstance(mate, MatePin):
                continue
            for name, host, host_pin, side in (
                (mate.from_name, mate.to_name, mate.to_pin, "l"),
                (mate.to_name, mate.from_name, mate.from_pin, "r"),
            ):
                if not self.connectors[name].attach:
                    continue
                host_connector = self.connectors[host]
                if host_connector.attach or host_connector.style == "simple":
                    raise Exception(
                        f"{name} can only be attached to a pin of a connector that is not simple"
                    )
                if name in attached_ports:
                    raise Exception(f"{name} is attached to more than one pin")
                pinindex = host_connector.pins.index(host_pin)
                attached = attachments.setdefault((host, pinindex, side), [])
                attached_ports[name] = f"{host}:p{pinindex+1}{side}a{len(attached)}"
                attached.append(name)
        for name, connector in self.connectors.items():
            if connector.attach and name not in attached_ports:
                raise Exception(f"{name} has attach set, but is not mated with any pin")

        def attachment_cell(host: str, pinindex: int, side: str) -> str:
            """HTML cell holding the connectors attached to one side of a pin."""
            attached = attachments.get((host, pinindex, side), [])
            if not attached:
                return '<td border="0"></td>'
            # short dashed line between attachment and pin, like a mate arrow
            dash = '<td border="0" width="3" height="1" fixedsize="true" bgcolor="#000000"></td>'
            gap = '<td border="0" width="2" height="1" fixedsize="true"></td>'
            dashes = dash + gap + dash + gap + dash
            line = (
                '<td border="0" cellpadding="0" valign="middle">'
                f'<table border="0" cellspacing="0" cellpadding="0"><tr>{dashes}</tr></table>'
                "</td>"
            )
            rows = []
            for k, name in enumerate(attached):
                box = (
                    f'<td border="1" cellpadding="3" port="p{pinindex+1}{side}a{k}">'
                    f"{html_line_breaks(self.connectors[name].type) or name}</td>"
                )
                rows.append(f"<tr>{box + line if side == 'l' else line + box}</tr>")
            spacer = '<tr><td colspan="2" border="0" height="2"></td></tr>'
            inner = (
                '<table border="0" cellspacing="0" cellpadding="0">'
                + spacer.join(rows)
                + "</table>"
            )
            return f'<td border="0" cellpadding="0">{inner}</td>'

        def node_port(name: str, port_str: str) -> str:
            """Graphviz node:port for a connector, resolving attached connectors."""
            return attached_ports.get(name, f"{name}{port_str}")

        for connector in self.connectors.values():
            if connector.attach:
                continue  # drawn inside the pin row of its host connector
            # If no wires connected (except maybe loop wires)?
            if not (connector.ports_left or connector.ports_right):
                connector.ports_left = True  # Use left side pins.

            html = []
            # fmt: off
            rows = [[f'{html_bgcolor(connector.bgcolor_title)}{remove_links(connector.name)}'
                        if connector.show_name else None],
                    [pn_info_string(HEADER_PN, None, remove_links(connector.pn)),
                     html_line_breaks(pn_info_string(HEADER_MPN, connector.manufacturer, connector.mpn)),
                     html_line_breaks(pn_info_string(HEADER_SPN, connector.supplier, connector.spn))
                        if self.options.show_supplier_in_diagram else None],
                    [html_line_breaks(connector.type),
                     html_line_breaks(connector.subtype),
                     f'{connector.pincount}-pin' if connector.show_pincount else None,
                     translate_color(connector.color, self.options.color_mode) if connector.color else None,
                     html_colorbar(connector.color)],
                    '<!-- connector table -->' if connector.style != 'simple' else None,
                    [html_image(connector.image)],
                    [html_caption(connector.image)]]
            # fmt: on

            rows.extend(get_additional_component_table(self, connector))
            rows.append([html_line_breaks(connector.notes)])
            attach_left = any(
                host == connector.name and side == "l" for host, _, side in attachments
            )
            attach_right = any(
                host == connector.name and side == "r" for host, _, side in attachments
            )
            attached_to = attach_left or attach_right

            # one list of cells per visible pin
            pin_rows = []
            if connector.style != "simple":
                for pinindex, (pinname, pinlabel, pincolor) in enumerate(
                    zip_longest(
                        connector.pins, connector.pinlabels, connector.pincolors
                    )
                ):
                    if (
                        connector.hide_disconnected_pins
                        and not connector.visible_pins.get(pinname, False)
                    ):
                        continue

                    cells = []
                    if connector.ports_left:
                        cells.append(f'<td port="p{pinindex+1}l">{pinname}</td>')
                    if pinlabel:
                        cells.append(f"<td>{pinlabel}</td>")
                    elif attached_to and connector.pinlabels:
                        cells.append("<td></td>")  # keep columns aligned
                    if connector.pincolors:
                        if pincolor in wv_colors._color_hex.keys():
                            # fmt: off
                            cells.append(f'<td sides="tbl">{translate_color(pincolor, self.options.color_mode)}</td>')
                            cells.append( '<td sides="tbr">'
                                          '<table border="0" cellborder="1"><tr>'
                                         f'<td bgcolor="{wv_colors.translate_color(pincolor, "HEX")}" width="8" height="8" fixedsize="true"></td>'
                                          '</tr></table>'
                                          '</td>')
                            # fmt: on
                        else:
                            cells.append('<td colspan="2"></td>')
                    if connector.ports_right:
                        cells.append(f'<td port="p{pinindex+1}r">{pinname}</td>')
                    pin_rows.append((pinindex, cells))

            if not attached_to:
                html.extend(
                    nested_html_table(rows, html_bgcolor_attr(connector.bgcolor))
                )
                if connector.style != "simple":
                    pinhtml = [
                        '<table border="0" cellspacing="0" cellpadding="3" cellborder="1">'
                    ]
                    for _, cells in pin_rows:
                        pinhtml.append("   <tr>")
                        pinhtml.extend(f"    {cell}" for cell in cells)
                        pinhtml.append("   </tr>")
                    pinhtml.append("  </table>")

                    if len(pinhtml) == 2:  # Table start and end with no rows between?
                        pinhtml = ["<!-- all pins hidden -->"]  # Avoid Graphviz error

                    html = [
                        row.replace("<!-- connector table -->", "\n".join(pinhtml))
                        for row in html
                    ]
            else:
                # Pin rows go directly into the outer table, with the attached
                # connectors in extra columns left/right of the part's outline.
                ncols = (
                    int(connector.ports_left)
                    + int(bool(connector.pinlabels))
                    + 2 * int(bool(connector.pincolors))
                    + int(connector.ports_right)
                )
                empty = '<td border="0"></td>'
                left = empty if attach_left else ""
                right = empty if attach_right else ""
                html.append(
                    '<table border="0" cellspacing="0" cellpadding="3" cellborder="1"'
                    f"{html_bgcolor_attr(connector.bgcolor)}>"
                )
                for row in rows:
                    if row == "<!-- connector table -->":
                        for pinindex, cells in pin_rows:
                            html.append(" <tr>")
                            if attach_left:
                                html.append(
                                    f"  {attachment_cell(connector.name, pinindex, 'l')}"
                                )
                            html.extend(f"  {cell}" for cell in cells)
                            if attach_right:
                                html.append(
                                    f"  {attachment_cell(connector.name, pinindex, 'r')}"
                                )
                            html.append(" </tr>")
                    elif row is not None and (not isinstance(row, list) or any(row)):
                        inner = "\n".join(nested_html_table([row]))
                        html.append(
                            f' <tr>{left}<td colspan="{ncols}" border="0" cellpadding="0">'
                            f"{inner}</td>{right}</tr>"
                        )
                html.append("</table>")

            html = "\n".join(html)
            dot.node(
                connector.name,
                label=f"<\n{html}\n>",
                # with attachments, the outline is drawn by the cell borders only
                shape="none" if attached_to else "box",
                style="filled",
                fillcolor=translate_color(self.options.bgcolor_connector, "HEX"),
                **self.buy_link(connector.name, connector.spn),
            )

            if len(connector.loops) > 0:
                dot.attr("edge", color="#000000:#ffffff:#000000")
                if connector.ports_left:
                    loop_side = "l"
                    loop_dir = "w"
                elif connector.ports_right:
                    loop_side = "r"
                    loop_dir = "e"
                else:
                    raise Exception("No side for loops")
                for loop in connector.loops:
                    dot.edge(
                        f"{connector.name}:p{loop[0]}{loop_side}:{loop_dir}",
                        f"{connector.name}:p{loop[1]}{loop_side}:{loop_dir}",
                        label=" ",  # Work-around to avoid over-sized loops.
                    )

        # determine if there are double- or triple-colored wires in the harness;
        # if so, pad single-color wires to make all wires of equal thickness
        pad = any(
            len(colorstr) > 2
            for cable in self.cables.values()
            for colorstr in cable.colors
        )

        for cable in self.cables.values():
            html = []

            awg_fmt = ""
            if cable.show_equiv:
                # Only convert units we actually know about, i.e. currently
                # mm2 and awg --- other units _are_ technically allowed,
                # and passed through as-is.
                if cable.gauge_unit == "mm\u00B2":
                    awg_fmt = f" ({awg_equiv(cable.gauge)} AWG)"
                elif cable.gauge_unit.upper() == "AWG":
                    awg_fmt = f" ({mm2_equiv(cable.gauge)} mm\u00B2)"

            # fmt: off
            rows = [[f'{html_bgcolor(cable.bgcolor_title)}{remove_links(cable.name)}'
                        if cable.show_name else None],
                    [pn_info_string(HEADER_PN, None,
                        remove_links(cable.pn)) if not isinstance(cable.pn, list) else None,
                     html_line_breaks(pn_info_string(HEADER_MPN,
                        cable.manufacturer if not isinstance(cable.manufacturer, list) else None,
                        cable.mpn if not isinstance(cable.mpn, list) else None)),
                     html_line_breaks(pn_info_string(HEADER_SPN,
                        cable.supplier if not isinstance(cable.supplier, list) else None,
                        cable.spn if not isinstance(cable.spn, list) else None))
                        if self.options.show_supplier_in_diagram else None],
                    [html_line_breaks(cable.type),
                     f'{cable.wirecount}x' if cable.show_wirecount else None,
                     f'{cable.gauge} {cable.gauge_unit}{awg_fmt}' if cable.gauge else None,
                     '+ S' if cable.shield else None,
                     f'{cable.length} {cable.length_unit}' if cable.length > 0 else None,
                     translate_color(cable.color, self.options.color_mode) if cable.color else None,
                     html_colorbar(cable.color)],
                    '<!-- wire table -->',
                    [html_image(cable.image)],
                    [html_caption(cable.image)]]
            # fmt: on

            rows.extend(get_additional_component_table(self, cable))
            rows.append([html_line_breaks(cable.notes)])
            html.extend(nested_html_table(rows, html_bgcolor_attr(cable.bgcolor)))

            wirehtml = []
            # conductor table
            wirehtml.append('<table border="0" cellspacing="0" cellborder="0">')
            wirehtml.append("   <tr><td>&nbsp;</td></tr>")

            for i, (connection_color, wirelabel) in enumerate(
                zip_longest(cable.colors, cable.wirelabels), 1
            ):
                wirehtml.append("   <tr>")
                wirehtml.append(f"    <td><!-- {i}_in --></td>")
                wirehtml.append(f"    <td>")

                wireinfo = []
                if cable.show_wirenumbers:
                    wireinfo.append(str(i))
                colorstr = wv_colors.translate_color(
                    connection_color, self.options.color_mode
                )
                if colorstr:
                    wireinfo.append(colorstr)
                if cable.wirelabels:
                    wireinfo.append(wirelabel if wirelabel is not None else "")
                wirehtml.append(f'     {":".join(wireinfo)}')

                wirehtml.append(f"    </td>")
                wirehtml.append(f"    <td><!-- {i}_out --></td>")
                wirehtml.append("   </tr>")

                # fmt: off
                bgcolors = ['#000000'] + get_color_hex(connection_color, pad=pad) + ['#000000']
                wirehtml.append(f"   <tr>")
                wirehtml.append(f'    <td colspan="3" border="0" cellspacing="0" cellpadding="0" port="w{i}" height="{(bar * len(bgcolors))}">')
                wirehtml.append('     <table cellspacing="0" cellborder="0" border="0">')
                for j, bgcolor in enumerate(bgcolors[::-1]):  # Reverse to match the curved wires when more than 2 colors
                    wirehtml.append(f'      <tr><td colspan="3" cellpadding="0" height="{bar}" bgcolor="{bgcolor if bgcolor != "" else wv_colors.default_color}" border="0"></td></tr>')
                wirehtml.append("     </table>")
                wirehtml.append("    </td>")
                wirehtml.append("   </tr>")
                # fmt: on

                # for bundles, individual wires can have part information
                if cable.category == "bundle":
                    # create a list of wire parameters
                    wireidentification = []
                    if isinstance(cable.pn, list):
                        wireidentification.append(
                            pn_info_string(
                                HEADER_PN, None, remove_links(cable.pn[i - 1])
                            )
                        )
                    manufacturer_info = pn_info_string(
                        HEADER_MPN,
                        (
                            cable.manufacturer[i - 1]
                            if isinstance(cable.manufacturer, list)
                            else None
                        ),
                        cable.mpn[i - 1] if isinstance(cable.mpn, list) else None,
                    )
                    supplier_info = pn_info_string(
                        HEADER_SPN,
                        (
                            cable.supplier[i - 1]
                            if isinstance(cable.supplier, list)
                            else None
                        ),
                        cable.spn[i - 1] if isinstance(cable.spn, list) else None,
                    )
                    if manufacturer_info:
                        wireidentification.append(html_line_breaks(manufacturer_info))
                    if supplier_info and self.options.show_supplier_in_diagram:
                        wireidentification.append(html_line_breaks(supplier_info))
                    # print parameters into a table row under the wire
                    if len(wireidentification) > 0:
                        # fmt: off
                        wirehtml.append('   <tr><td colspan="3">')
                        wirehtml.append('    <table border="0" cellspacing="0" cellborder="0"><tr>')
                        for attrib in wireidentification:
                            wirehtml.append(f"     <td>{attrib}</td>")
                        wirehtml.append("    </tr></table>")
                        wirehtml.append("   </td></tr>")
                        # fmt: on

            if cable.shield:
                wirehtml.append("   <tr><td>&nbsp;</td></tr>")  # spacer
                wirehtml.append("   <tr>")
                wirehtml.append("    <td><!-- s_in --></td>")
                wirehtml.append("    <td>Shield</td>")
                wirehtml.append("    <td><!-- s_out --></td>")
                wirehtml.append("   </tr>")
                if isinstance(cable.shield, str):
                    # shield is shown with specified color and black borders
                    shield_color_hex = wv_colors.get_color_hex(cable.shield)[0]
                    attributes = (
                        f'height="6" bgcolor="{shield_color_hex}" border="2" sides="tb"'
                    )
                else:
                    # shield is shown as a thin black wire
                    attributes = f'height="2" bgcolor="#000000" border="0"'
                # fmt: off
                wirehtml.append(f'   <tr><td colspan="3" cellpadding="0" {attributes} port="ws"></td></tr>')
                # fmt: on

            wirehtml.append("   <tr><td>&nbsp;</td></tr>")
            wirehtml.append("  </table>")

            html = [
                row.replace("<!-- wire table -->", "\n".join(wirehtml)) for row in html
            ]

            edge_weight = {"weight": str(cable.weight)} if cable.weight else {}

            # connections
            for connection in cable.connections:
                if isinstance(connection.via_port, int):
                    # check if it's an actual wire and not a shield
                    dot.attr(
                        "edge",
                        color=":".join(
                            ["#000000"]
                            + wv_colors.get_color_hex(
                                cable.colors[connection.via_port - 1], pad=pad
                            )
                            + ["#000000"]
                        ),
                    )
                else:  # it's a shield connection
                    # shield is shown with specified color and black borders, or as a thin black wire otherwise
                    dot.attr(
                        "edge",
                        color=(
                            ":".join(["#000000", shield_color_hex, "#000000"])
                            if isinstance(cable.shield, str)
                            else "#000000"
                        ),
                    )
                if connection.from_pin is not None:  # connect to left
                    from_connector = self.connectors[connection.from_name]
                    from_pin_index = from_connector.pins.index(connection.from_pin)
                    from_port_str = (
                        f":p{from_pin_index+1}r"
                        if from_connector.style != "simple"
                        else ""
                    )
                    code_left_1 = f"{node_port(connection.from_name, from_port_str)}:e"
                    code_left_2 = f"{cable.name}:w{connection.via_port}:w"
                    dot.edge(code_left_1, code_left_2, **edge_weight)
                    if from_connector.show_name:
                        from_info = [
                            str(connection.from_name),
                            str(connection.from_pin),
                        ]
                        if from_connector.pinlabels:
                            pinlabel = from_connector.pinlabels[from_pin_index]
                            if pinlabel != "":
                                from_info.append(pinlabel)
                        from_string = ":".join(from_info)
                    else:
                        from_string = ""
                    html = [
                        row.replace(f"<!-- {connection.via_port}_in -->", from_string)
                        for row in html
                    ]
                if connection.to_pin is not None:  # connect to right
                    to_connector = self.connectors[connection.to_name]
                    to_pin_index = to_connector.pins.index(connection.to_pin)
                    to_port_str = (
                        f":p{to_pin_index+1}l" if to_connector.style != "simple" else ""
                    )
                    code_right_1 = f"{cable.name}:w{connection.via_port}:e"
                    code_right_2 = f"{node_port(connection.to_name, to_port_str)}:w"
                    dot.edge(code_right_1, code_right_2, **edge_weight)
                    if to_connector.show_name:
                        to_info = [str(connection.to_name), str(connection.to_pin)]
                        if to_connector.pinlabels:
                            pinlabel = to_connector.pinlabels[to_pin_index]
                            if pinlabel != "":
                                to_info.append(pinlabel)
                        to_string = ":".join(to_info)
                    else:
                        to_string = ""
                    html = [
                        row.replace(f"<!-- {connection.via_port}_out -->", to_string)
                        for row in html
                    ]

            style, bgcolor = (
                ("filled,dashed", self.options.bgcolor_bundle)
                if cable.category == "bundle"
                else ("filled", self.options.bgcolor_cable)
            )
            html = "\n".join(html)
            dot.node(
                cable.name,
                label=f"<\n{html}\n>",
                shape="box",
                style=style,
                fillcolor=translate_color(bgcolor, "HEX"),
                **self.buy_link(cable.name, cable.spn),
            )

        # mates
        for mate in self.mates:
            if mate.from_name in attached_ports or mate.to_name in attached_ports:
                continue  # drawn as attachment, see above
            if mate.shape[-1] == ">":
                dir = "both" if mate.shape[0] == "<" else "forward"
            else:
                dir = "back" if mate.shape[0] == "<" else "none"

            if isinstance(mate, MatePin):
                color = "#000000"
            elif isinstance(mate, MateComponent):
                color = "#000000:#000000"
            else:
                raise Exception(f"{mate} is an unknown mate")

            from_connector = self.connectors[mate.from_name]
            to_connector = self.connectors[mate.to_name]
            if isinstance(mate, MatePin) and from_connector.style != "simple":
                from_pin_index = from_connector.pins.index(mate.from_pin)
                from_port_str = f":p{from_pin_index+1}r"
            else:  # MateComponent or style == 'simple'
                from_port_str = ""
            if isinstance(mate, MatePin) and to_connector.style != "simple":
                to_pin_index = to_connector.pins.index(mate.to_pin)
                to_port_str = f":p{to_pin_index+1}l"
            else:  # MateComponent or style == 'simple'
                to_port_str = ""
            code_from = f"{mate.from_name}{from_port_str}:e"
            code_to = f"{mate.to_name}{to_port_str}:w"

            dot.attr("edge", color=color, style="dashed", dir=dir)
            dot.edge(code_from, code_to)

        def typecheck(name: str, value: Any, expect: type) -> None:
            if not isinstance(value, expect):
                raise Exception(
                    f"Unexpected value type of {name}: Expected {expect}, got {type(value)}\n{value}"
                )

        # TODO?: Differ between override attributes and HTML?
        if self.tweak.override is not None:
            typecheck("tweak.override", self.tweak.override, dict)
            for k, d in self.tweak.override.items():
                typecheck(f"tweak.override.{k} key", k, str)
                typecheck(f"tweak.override.{k} value", d, dict)
                for a, v in d.items():
                    typecheck(f"tweak.override.{k}.{a} key", a, str)
                    typecheck(f"tweak.override.{k}.{a} value", v, (str, type(None)))

            # Override generated attributes of selected entries matching tweak.override.
            for i, entry in enumerate(dot.body):
                if isinstance(entry, str):
                    # Find a possibly quoted keyword after leading TAB(s) and followed by [ ].
                    match = re.match(
                        r'^\t*(")?((?(1)[^"]|[^ "])+)(?(1)") \[.*\]$', entry, re.S
                    )
                    keyword = match and match[2]
                    if keyword in self.tweak.override.keys():
                        for attr, value in self.tweak.override[keyword].items():
                            if value is None:
                                entry, n_subs = re.subn(
                                    f'( +)?{attr}=("[^"]*"|[^] ]*)(?(1)| *)', "", entry
                                )
                                if n_subs < 1:
                                    print(
                                        f"Harness.create_graph() warning: {attr} not found in {keyword}!"
                                    )
                                elif n_subs > 1:
                                    print(
                                        f"Harness.create_graph() warning: {attr} removed {n_subs} times in {keyword}!"
                                    )
                                continue

                            if len(value) == 0 or " " in value:
                                value = value.replace('"', r"\"")
                                value = f'"{value}"'
                            entry, n_subs = re.subn(
                                f'{attr}=("[^"]*"|[^] ]*)', f"{attr}={value}", entry
                            )
                            if n_subs < 1:
                                # If attr not found, then append it
                                entry = re.sub(r"\]$", f" {attr}={value}]", entry)
                            elif n_subs > 1:
                                print(
                                    f"Harness.create_graph() warning: {attr} overridden {n_subs} times in {keyword}!"
                                )

                        dot.body[i] = entry

        if self.tweak.append is not None:
            if isinstance(self.tweak.append, list):
                for i, element in enumerate(self.tweak.append, 1):
                    typecheck(f"tweak.append[{i}]", element, str)
                dot.body.extend(self.tweak.append)
            else:
                typecheck("tweak.append", self.tweak.append, str)
                dot.body.append(self.tweak.append)

        # Tweak processing above must be the last before returning dot.
        # Please don't insert any code that might change the dot contents
        # after tweak processing.

        return dot

    def buy_link(self, designator: str, spn: Any) -> dict:
        """
        Node attributes making a diagram node clickable, when its supplier part
        number (or that of an additional BOM item listing it) is a URL.
        """
        urls = [spn] + [
            item.get("spn")
            for item in self.additional_bom_items
            if designator in (item.get("designators") or [])
        ]
        for url in urls:
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return {"URL": url, "target": "_blank", "tooltip": f"Buy {designator}"}
        return {}

    def bom_graph(self) -> Graph:
        """The BOM as a single table node, with clickable supplier links."""
        from html import escape

        rows = bom_list(self.bom())
        header, entries = rows[0], rows[1:]
        title = escape(str(self.metadata.get("title", "")))
        font = f'face="{self.options.fontname}"'
        html = ['<table border="0" cellborder="1" cellspacing="0" cellpadding="4">']
        html.append(
            f'<tr><td colspan="{len(header)}" border="0" align="left">'
            f'<font {font} point-size="16"><b>{title}</b> – Bill of Materials</font></td></tr>'
        )
        html.append(
            "<tr>"
            + "".join(f'<td bgcolor="#eeeeee"><b>{escape(h)}</b></td>' for h in header)
            + "</tr>"
        )
        spn_col = header.index(HEADER_SPN) if HEADER_SPN in header else None
        nodes = set(self.connectors) | set(self.cables)

        def designator_links(value: str) -> str:
            # each designator links to its node in the diagram (see output_pdf)
            parts = [
                f'<td href="#wireviz-node:{escape(d)}" border="0" cellpadding="0">'
                f'<font color="#0645ad"><u>{escape(d)}</u></font></td>'
                if d in nodes
                else f'<td border="0" cellpadding="0">{escape(d)}</td>'
                for d in value.split(", ")
            ]
            sep = '<td border="0" cellpadding="0">, </td>'
            return (
                '<td align="left"><table border="0" cellspacing="0" cellpadding="0"><tr>'
                + sep.join(parts)
                + "</tr></table></td>"
            )
        for entry in entries:
            url = entry[spn_col] if spn_col is not None else ""
            is_url = url.startswith(("http://", "https://"))
            cells = []
            for i, value in enumerate(entry):
                text = escape(value)
                if is_url and i == spn_col:
                    cells.append(
                        f'<td align="left" href="{escape(url)}" target="_blank">'
                        f'<font color="#0645ad"><u>{text}</u></font></td>'
                    )
                elif is_url and header[i] == "Description":
                    cells.append(
                        f'<td align="left" href="{escape(url)}" target="_blank">'
                        f'<font color="#0645ad">{text}</font></td>'
                    )
                elif header[i] == "Designators" and value:
                    cells.append(designator_links(value))
                else:
                    cells.append(f'<td align="left">{text}</td>')
            html.append("<tr>" + "".join(cells) + "</tr>")
        html.append("</table>")
        bom = Graph()
        bom.attr("graph", fontname=self.options.fontname, margin="0.5")
        bom.attr("node", shape="none", fontname=self.options.fontname)
        bom.node("BOM", label="<" + "".join(html) + ">")
        return bom

    def output_pdf(self, filename: (str, Path), source, render_kwargs: dict) -> None:
        """PDF with the diagram on the first page and the BOM on the second."""
        from pypdf import PdfWriter

        diagram = source.render(
            filename=f"{filename}.diagram.tmp", format="pdf", cleanup=True, **render_kwargs
        )
        bom = self.bom_graph().render(
            filename=f"{filename}.bom.tmp", format="pdf", cleanup=True
        )
        writer = PdfWriter()
        for part in (diagram, bom):
            writer.append(part)
        self.link_designators(writer, source, render_kwargs)
        writer.add_metadata({"/Title": str(self.metadata.get("title", ""))})
        with open(f"{filename}.pdf", "wb") as f:
            writer.write(f)
        for part in (diagram, bom):
            Path(part).unlink()

    def link_designators(self, writer, source, render_kwargs: dict) -> None:
        """
        Turn the `#wireviz-node:<designator>` links of the BOM page into links
        that jump to the diagram page, zoomed to that node.
        """
        import json

        from pypdf.generic import (
            ArrayObject,
            DictionaryObject,
            FloatObject,
            NameObject,
        )

        layout = json.loads(source.pipe(format="json", **render_kwargs))
        bb = [float(v) for v in layout["bb"].split(",")]
        diagram = writer.pages[0]
        box = diagram.mediabox
        # Graphviz centers the drawing on the page with a margin
        pad_x = (float(box.width) - (bb[2] - bb[0])) / 2 + float(box.left)
        pad_y = (float(box.height) - (bb[3] - bb[1])) / 2 + float(box.bottom)
        rects = {}
        for obj in layout.get("objects", []):
            if "pos" not in obj or "width" not in obj:
                continue
            x, y = (float(v) for v in obj["pos"].split(","))
            w, h = float(obj["width"]) * 72, float(obj["height"]) * 72
            margin = 36  # some space around the node when zoomed in
            rects[obj["name"]] = [
                x - w / 2 - bb[0] + pad_x - margin,
                y - h / 2 - bb[1] + pad_y - margin,
                x + w / 2 - bb[0] + pad_x + margin,
                y + h / 2 - bb[1] + pad_y + margin,
            ]

        for page in writer.pages[1:]:
            for annot in page.get("/Annots", []):
                annot = annot.get_object()
                action = annot.get("/A")
                uri = action.get_object().get("/URI") if action else None
                if not uri or not str(uri).startswith("#wireviz-node:"):
                    continue
                rect = rects.get(str(uri).split(":", 1)[1])
                if rect is None:
                    continue
                destination = ArrayObject(
                    [diagram.indirect_reference, NameObject("/FitR")]
                    + [FloatObject(v) for v in rect]
                )
                annot[NameObject("/A")] = DictionaryObject(
                    {
                        NameObject("/S"): NameObject("/GoTo"),
                        NameObject("/D"): destination,
                    }
                )

    # cache for the GraphViz Graph object
    # do not access directly, use self.graph instead
    _graph = None

    @property
    def graph(self):
        if not self._graph:  # no cached graph exists, generate one
            self._graph = self.create_graph()
        return self._graph  # return cached graph

    def render_source(self):
        """
        Return the object to render (graph or laid-out source) and the extra
        render arguments. With options.smooth_wires, dot only places the nodes;
        every edge is then replaced by a single cubic Bezier curve with
        horizontal tangents, and neato renders the result without re-layout.
        """
        if not self.options.smooth_wires:
            return self.graph, {}
        layout = self.graph.pipe(format="dot", encoding="utf-8")
        return (
            Source(smooth_edges(layout, self.options.wire_width), engine="neato"),
            {"neato_no_op": 2},
        )

    @property
    def png(self):
        from io import BytesIO

        source, kwargs = self.render_source()
        data = BytesIO()
        data.write(source.pipe(format="png", **kwargs))
        data.seek(0)
        return data.read()

    @property
    def svg(self):  # TODO?: Verify xml encoding="utf-8" in SVG?
        source, kwargs = self.render_source()
        return embed_svg_images(
            source.pipe(format="svg", **kwargs).decode("utf-8"), Path.cwd()
        )

    def output(
        self,
        filename: (str, Path),
        view: bool = False,
        cleanup: bool = True,
        fmt: tuple = ("html", "png", "svg", "tsv"),
    ) -> None:
        # graphical output
        graph = self.graph
        source, render_kwargs = self.render_source()
        svg_already_exists = Path(
            f"{filename}.svg"
        ).exists()  # if SVG already exists, do not delete later
        # graphical output
        for f in fmt:
            if f in ("png", "svg", "html"):
                if f == "html":  # if HTML format is specified,
                    f = "svg"  # generate SVG for embedding into HTML
                # SVG file will be renamed/deleted later
                _filename = f"{filename}.tmp" if f == "svg" else filename
                # TODO: prevent rendering SVG twice when both SVG and HTML are specified
                source.render(
                    filename=_filename,
                    format=f,
                    view=view,
                    cleanup=cleanup,
                    **render_kwargs,
                )
        # embed images into SVG output
        if "svg" in fmt or "html" in fmt:
            embed_svg_images_file(f"{filename}.tmp.svg")
        # GraphViz output
        if "gv" in fmt:
            graph.save(filename=f"{filename}.gv")
        # BOM output
        bomlist = bom_list(self.bom())
        if "tsv" in fmt:
            file_write_text(f"{filename}.bom.tsv", tuplelist2tsv(bomlist))
        if "csv" in fmt:
            # TODO: implement CSV output (preferrably using CSV library)
            print("CSV output is not yet supported")
        # HTML output
        if "html" in fmt:
            generate_html_output(filename, bomlist, self.metadata, self.options)
        # PDF output
        if "pdf" in fmt:
            self.output_pdf(filename, source, render_kwargs)
        # delete SVG if not needed
        if "html" in fmt and not "svg" in fmt:
            # SVG file was just needed to generate HTML
            Path(f"{filename}.tmp.svg").unlink()
        elif "svg" in fmt:
            Path(f"{filename}.tmp.svg").replace(f"{filename}.svg")

    def bom(self):
        if not self._bom:
            self._bom = generate_bom(self)
        return self._bom
