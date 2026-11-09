import torch


def sanitize_for_dot(obj):
    """Escapes characters that break Graphviz record labels."""
    s = str(obj)
    # Remove literal newlines that break the label string
    s = s.replace('\n', ' ').replace('\r', '')

    # Escape brackets so Graphviz doesn't think they are ports
    s = s.replace('<', '\\<').replace('>', '\\>')

    # Escape curly braces just in case they appear in the tensor printout
    s = s.replace('{', '\\{').replace('}', '\\}')

    # Optional: truncate if strings are insanely long
    if len(s) > 500:
        s = s[:497] + "..."

    return s

def aten_to_dot(gm):
    from torch.fx.passes.graph_drawer import FxGraphDrawer
    dot = FxGraphDrawer(gm, "f").get_dot_graph()
    for node in gm.graph.nodes:
        if node.op == 'call_function' and isinstance(node.target, torch._ops.OpOverload):# and node.target._overloadpacket in supported_ops:
            # print('Node name', node.name)
            col = "{"
            if 'flops' in node.meta:
                count = node.meta['flops']
                col += "|" + "Flops" + "=" + sanitize_for_dot(str(count)) + r"\n"
                # dot_node = dot.get_node(node.name)[0]
                # dot_node.set_label(f'{dot_node.get_label()}|Flops: {count}')

            if 'input_size' in node.meta:
                inputs = node.meta['input_size']
                # print('inputs', node, str(inputs))
                # dot_node = dot.get_node(node.name)[0]
                # dot_node.set_label(f'{dot_node.get_label()}|Inputs: {inputs}')
                inputs = [',\n'.join([str(input) for input in inputs])]
                col += "|" + "Inputs" + "=" + sanitize_for_dot(str(inputs)) + r"\n"

            if 'output_size' in node.meta:
                outputs = node.meta['output_size']
                # dot_node = dot.get_node(node.name)[0]
                # dot_node.set_label(f'{dot_node.get_label()}|Outputs: {outputs}\nmm')
                col += "|" + "Outputs" + "=" + sanitize_for_dot(str(outputs)) + r"\n"

            if 'flop_type' in node.meta:
                flop_type = node.meta['flop_type']
                col += "|" + "Flop type" + "=" + flop_type + r"\n"
            if 'flop_req' in node.meta:
                col += "|" + sanitize_for_dot(str(node.meta['flop_req'])) + r"\n"

            col += '}'

            dot_node = dot.get_node(node.name)[0]
            # print(dot_node)
            dot_node.set_label(f'{dot_node.get_label()}|{col}')
            # print('Node label for', node.name, dot.get_node(node.name)[0])
            # print(dot_node)
        elif node.op in ['get_attr', 'placeholder']:
            col = "{"
            if 'output_size' in node.meta:
                outputs = node.meta['output_size']
                col += "|" + "Outputs" + "=" + sanitize_for_dot(str(outputs)) + r"\n"
            if 'flop_type' in node.meta:
                flop_type = node.meta['flop_type']
                col += "|" + "Flop type" + "=" + flop_type + r"\n"
            col += '}'

            dot_node = dot.get_node(node.name)[0]
            dot_node.set_label(f'{dot_node.get_label()}|{col}')

    return dot

def write_aten_svg(gm, filename='fx_flop.svg'):
    dot = aten_to_dot(gm)
    dot.write_raw(str(filename).rstrip('.svg') + '.dot')
    write_dot(dot, filename)

def write_dot(dot, filename):
    dot.write_svg(filename)
    dot.set_dpi(300)
    dot.write_png(str(filename).rstrip('.svg') + '.png')


# ---------------------------------------------------------------------------
# Region visualization
# ---------------------------------------------------------------------------


