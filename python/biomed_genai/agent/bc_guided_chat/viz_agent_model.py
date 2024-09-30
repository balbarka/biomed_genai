# Local discovery of agent app
def agent_model_graphic(config=None):
    import re
    from graphviz import Digraph

    dot = Digraph('pt')
    dot.attr(compound='true')
    dot.graph_attr['rankdir'] = 'LR'
    dot.edge_attr.update(arrowhead='none', arrowsize='1')
    dot.attr('node', shape='rectangle')

    dot.node('question', 'question', shape='oval',  fillcolor='#DAF7A6', style='filled', **{'width': "2"})
    
    with dot.subgraph(name='cluster_agent') as ca:
        ca.body.append(r'label="bc_guided_chat Agent"')
        ca.body.append('style="filled"')
        ca.body.append('color="#808080"')
        ca.body.append('fillcolor="#d6eaf8"')        

        ca.node('run_qcat', 'run_qcat', shape='rect',  fillcolor='#87CEEB', style='filled', **{'width': "2"})
        ca.node('run_retriever', 'run_retriever', shape='rect',  fillcolor='#87CEEB', style='filled', **{'width': "2"})
        ca.node('run_prompt', 'run_prompt', shape='rect',  fillcolor='#87CEEB', style='filled', **{'width': "2"})
        ca.node('run_answer', 'run_answer', shape='rect',  fillcolor='#87CEEB', style='filled', **{'width': "2"})

    dot.node('answer', 'answer', shape='oval',  fillcolor='#DAF7A6', style='filled', **{'width': "2"})

    dot.node('blank_ep1', 'blank_ep1', shape='rect',  style='invis', **{'width': "2"})
    dot.node('llama3_qcat', 'llama3_qcat', shape='rect',  fillcolor='#d6eaf8', style='filled', **{'width': "2"})
    dot.node('vs_retriever', 'vs_retriever', shape='rect',  fillcolor='#d6eaf8', style='filled', **{'width': "2"})
    dot.node('blank_ep4', 'blank_ep4', shape='rect', style='invis', **{'width': "2"})
    dot.node('llama3_answer', 'llama3_answer', shape='rect',  fillcolor='#d6eaf8', style='filled', **{'width': "2"})
    dot.node('blank_ep6', 'blank_ep6', shape='rect', style='invis', **{'width': "2"})

    dot.edge('question','run_qcat', arrowhead='normal')
    dot.edge('run_qcat','run_retriever', arrowhead='normal')
    dot.edge('run_retriever','run_prompt', arrowhead='normal')
    dot.edge('run_prompt','run_answer', arrowhead='normal')
    dot.edge('run_answer','answer', arrowhead='normal')

    dot.edge('blank_ep1','llama3_qcat', style='invis')
    dot.edge('llama3_qcat','vs_retriever', style='invis')
    dot.edge('vs_retriever','blank_ep4', style='invis')
    dot.edge('blank_ep4','llama3_answer', style='invis')
    dot.edge('llama3_answer','blank_ep6', style='invis')

    dot.edge('run_qcat','llama3_qcat', dir='both', arrowhead='normal')
    dot.edge('llama3_qcat', 'run_qcat')

    dot.edge('run_retriever','vs_retriever', dir='both', arrowhead='normal')
    dot.edge('vs_retriever', 'run_retriever')

    dot.edge('run_answer','llama3_answer', dir='both', arrowhead='normal')
    dot.edge('llama3_answer', 'run_answer')

    dot.edge('blank_ep1', 'run_qcat', style='invis')

    html = dot._repr_image_svg_xml()

    html = re.sub(r'<svg width=\"\d*pt\" height=\"\d*pt\"',
                  '<div style="text-align:center;"><svg width="800pt" aligned=center', html)

    html = re.sub(r'stroke-width=\"2\"', 'stroke-width=\"4\"', html)

    return html