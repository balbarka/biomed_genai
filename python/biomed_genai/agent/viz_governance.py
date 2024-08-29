# This Viz displays the agent inner-loop and out-loop governance as notebook-tasks

def agent_governance_graphic(PROJECT_ROOT_PATH, curr_nb_path: str = ""):
    import re
    from graphviz import Digraph

    url_root_path = "#w" + PROJECT_ROOT_PATH[2:] + '/databricks/agent/bc_qa_chat/' 
    curr_nb_name = curr_nb_path.split('/')[-1]


    # We are going to create notebook URLS using PROJECT_ROOT_PATH and notebook name
    def nb_node(cluster, name, tgt_nb_path, fillcolor='#99CCFF'):
        tgt_nb_name = tgt_nb_path.split('/')[-1]
        fillcolor = '#FFFF00' if tgt_nb_name==curr_nb_name else fillcolor
        cluster.node(name, label=tgt_nb_name, fillcolor=fillcolor, style='filled', shape='rect',
                     **{'width': "4", 'tooltip': 'notebook', 'href': url_root_path + tgt_nb_path, 'target': "_blank"})
                
    # Because this graphic can be called before some entities have been created
    # we'll conditionally populate labels / href depending on their existance

    dot = Digraph('pt')
    dot.attr(compound='true')
    dot.graph_attr['rankdir'] = 'TB'
    dot.edge_attr.update(arrowhead='none', arrowsize='0.5')
    dot.attr('edge', penwidth='2')
    dot.attr('node', shape='rectangle')
    dot.attr(splines='ortho')

    with dot.subgraph(name='cluster_bp_workflow') as bp:
        bp.body.append(r'label="Agent Outer-Loop & Inner-Loop Notebook-Task Workflow"')

        with bp.subgraph(name='cluster_outer_loop') as ol:
            ol.body.append(r'label=""')
            ol.attr(style='invisible') 

            nb_node(ol, 'eval_ds', '01_DATASET_bc_eval_ds')
            nb_node(ol, 'config', '00_CONFIG_bc_qa_chat_config')
            ol.node('ol_blank', label='', shape='point', width='0', height='0')
            nb_node(ol, 'prod_release', '05_RELEASE_biomed_genai')
            nb_node(ol, 'dashboard', '04_DASHBOARDS')
            nb_node(ol, 'monitor_metrics', '03_MONITOR_bc_qa_chat_metrics')

        with bp.subgraph(name='cluster_inner_loop') as il:
            il.body.append(r'label="{agent_models}"')
            il.body.append('style="filled"')
            il.body.append('color="black"')
            il.body.append('fillcolor="#99CCFF"')
            il.graph_attr['rankdir'] = 'TB'


            #il.node('candidate_runs', 'Candidate Runs', shape='rect', style='filled', fillcolor='#87CEEB',
            #        **{'width': "4"})
            #nb_node(il, 'candidate_runs', 'agent_models/03_01_Candidate_Runs')
            with il.subgraph(name='cluster_candidate_runs') as cr:
                cr.body.append(r'label="{candidate_runs}"')
                cr.body.append('style="filled"')
                cr.body.append('color="black"')
                cr.body.append('fillcolor="#BBEEFF"')
                cr.graph_attr['rankdir'] = 'TB'

                cr.node('models', 'agent_model/models', shape='folder', fillcolor='#EEEAB6', style='filled',
                        **{'width': "3.78", 'tooltip': 'models',
                           'href': url_root_path + 'agent_model/models', 'target': "_blank"})

            nb_node(il, 'score_register', 'agent_model/02_02_Score_Register', '#BBEEFF')
            nb_node(il, 'review_app', 'agent_model/02_03_Review_App_Feedback', '#BBEEFF')
            nb_node(il, 'champion_challenger', 'agent_model/02_04_Champion_Challenger','#BBEEFF')

            with il.subgraph(name='cluster_return_loop') as rl:
                rl.body.append(r'label=""')
                rl.attr(style='invisible') 
                rl.node('rl_blank_1', label='', shape='point', width='0', height='0')
                rl.node('rl_blank_2', label='', shape='point', width='0', height='0')

    dot.edge('eval_ds','config', arrowhead='normal', dir='back')
    dot.edge('config','ol_blank', color='#BF40BF', arrowhead='normal', dir='back')
    dot.edge('ol_blank','prod_release', color='#BF40BF')
    dot.edge('prod_release', 'dashboard', arrowhead='normal', dir='back')
    dot.edge('dashboard', 'monitor_metrics', arrowhead='normal', dir='back')

    #dot.edge('eval_ds', 'candidate_runs', lhead='cluster_inner_loop', arrowhead='normal')
    dot.edge('eval_ds', 'models', lhead='cluster_inner_loop', arrowhead='normal')

    #dot.edge('candidate_runs', 'score_register', arrowhead='normal')
    dot.edge('models', 'score_register', arrowhead='normal')

    dot.edge('score_register', 'review_app', arrowhead='normal')
    dot.edge('review_app', 'champion_challenger', arrowhead='normal')

    #dot.edge('candidate_runs', 'rl_blank_1', color='#BF40BF', arrowhead='normal', dir='back')
    dot.edge('models', 'rl_blank_1', color='#BF40BF', arrowhead='normal', dir='back', ltail='cluster_candidate_runs')

    dot.edge('rl_blank_1', 'rl_blank_2', color='#BF40BF')

    dot.edge('rl_blank_2', 'champion_challenger', color='#BF40BF')

    dot.edge('champion_challenger', 'monitor_metrics', ltail='cluster_inner_loop', arrowhead='normal')

    html = dot._repr_image_svg_xml()

    html = dot._repr_image_svg_xml().format(
        agent_models=f'<a href="{url_root_path}02_AGENT_MODELS" target="_blank">02_AGENT_MODELS</a>',
        candidate_runs=f'<a href="{url_root_path}agent_model/02_01_Candidate_Runs" target="_blank">02_01_Candidate_Runs</a>')

    html = re.sub(r'<svg width=\"\d*pt\" height=\"\d*pt\"',
                  '<div style="text-align:center;"><svg width="600pt" aligned=center', html)

    html = re.sub(r'stroke-width=\"2\"', 'stroke-width=\"4\"', html)

    return html
