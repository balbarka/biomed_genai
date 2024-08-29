def agent_deploy_graphic(config):
    import re
    from graphviz import Digraph

    dot = Digraph('pt')
    dot.attr(compound='true')
    dot.graph_attr['rankdir'] = 'LR'
    dot.edge_attr.update(arrowhead='none', arrowsize='1')
    dot.attr('node', shape='rectangle')

    with dot.subgraph(name='cluster_agent_deployment') as ad:
        ad.body.append(r'label="{agent_deployment_label}"')

        with ad.subgraph(name='cluster_uc') as uc:
            uc.body.append(r'label="{unity_catalog_label}"')
            uc.body.append('style="filled"')
            uc.body.append('color="#808080"')
            uc.body.append('fillcolor="#F5F5F5"')

            uc.node('blank1', 'blank1', shape='rect', style='invis', **{'width': "2"})

            with uc.subgraph(name='cluster_uc_delta') as ucd:
                ucd.body.append(r'label="Delta (DataSet)"')
                ucd.body.append('style="filled"')
                ucd.body.append('color="#808080"')
                ucd.body.append('fillcolor="#e4f2f7"')

                ucd.node('eval_ds', config.experiment.eval_ds.table, fillcolor='#CAD9EF', style='filled', shape='rect',
                    **{'width': "2", 'tooltip': "Evaluation Dataset",
                       'href': config.experiment.eval_ds.uc_relative_url, 'target': "_blank"})

            with uc.subgraph(name='cluster_uc_model') as ucm:
                ucm.body.append(r'label="Registered Model"')
                ucm.body.append('style="filled"')
                ucm.body.append('color="#808080"')
                ucm.body.append('fillcolor="#e4f2f7"')

                ucm.node("registered_model", f'{config.registered_model.model_name} (v{config.release_version})',
                         fillcolor='#87CEEB', style='filled', shape='rect',
                         **{'width': "2", 'tooltip': "Registered Model",
                            'href': config.registered_model.uc_relative_url, 'target': "_blank"})
                
            with uc.subgraph(name='cluster_uc_inference') as uci:
                uci.body.append(r'label="Delta (Inference)"')
                uci.body.append('style="filled"')
                uci.body.append('color="#808080"')
                uci.body.append('fillcolor="#e4f2f7"')

                uci.node("inference", "Inference Table",
                         fillcolor='#87CEEB', style='filled', shape='rect',
                         **{'width': "2", 'tooltip': "Inference Table",
                            'href': config.registered_model.uc_relative_url, 'target': "_blank"})

            uc.node('blank2', 'blank2', shape='rect', style='invis', **{'width': "2"})

        with ad.subgraph(name='cluster_ws') as ws:
            ws.body.append('label="Workspace"')
            ws.body.append('labelloc="b"')
            ws.body.append('style="filled"')
            ws.body.append('color="#808080"')
            ws.body.append('fillcolor="#F5F5F5"')
            
            with ws.subgraph(name='cluster_nb') as nb:
                nb.body.append('label="Notebook (Model)"')
                nb.body.append('labelloc="b"')
                nb.body.append('style="filled"')
                nb.body.append('color="#808080"')
                nb.body.append('fillcolor="#e4f2f7"')

                nb.node('notebook_model', config.default_model_name or 'MODEL',
                    fillcolor='#87CEEB', style='filled', shape='rect',
                    **{'width': "1.78", 'tooltip': "Notebook Model"})

            with ws.subgraph(name='cluster_ws_experiment') as exp:
                exp.body.append(r'label="{experiment_label}"')
                exp.body.append('style="filled"')
                exp.body.append('color="#808080"')
                exp.body.append('fillcolor="#e4f2f7"')

                with exp.subgraph(name='cluster_ws_experiment_model') as mdl:
                    mdl.body.append(r'label="{model_run}"')
                    mdl.body.append('style="filled"')
                    mdl.body.append('color="#808080"')
                    mdl.body.append('fillcolor="#87CEEB"')
                
                    mdl.node("eval_run", config.experiment.eval_ds.ds_release_version_name,
                             fillcolor='#CAD9EF', style='filled', shape='rect',
                             **{'width': "2", 'tooltip': "Evaluation Dataset",
                            'href': config.experiment.ws_exp_release_relative_url, 'target': "_blank"})

            with ws.subgraph(name='cluster_ws_end_point') as ep:
                ep.body.append(r'label="{end_point}"')
                ep.body.append('style="filled"')
                ep.body.append('color="#808080"')
                ep.body.append('fillcolor="#e4f2f7"')

                ep.node("served_model","Agent Model", fillcolor='#87CEEB', style='filled', shape='rect',
                        **{'width': "2",
                           'tooltip': "TODO",
                           'href': "TODO",
                           'target': "_blank"})

                ep.node("feedback_agent","Agent Feedback", fillcolor='#87CEEB', style='filled', shape='rect',
                        **{'width': "2",
                           'tooltip': "TODO",
                           'href': "TODO",
                           'target': "_blank"})

            with ws.subgraph(name='cluster_ws_metrics') as wsm:
                wsm.body.append(r'label="Notebook (Metrics)"')
                wsm.body.append('style="filled"')
                wsm.body.append('color="#808080"')
                wsm.body.append('fillcolor="#e4f2f7"')

                wsm.node("metrics_nb","bc_qa_chat_metrics", fillcolor='#87CEEB', style='filled', shape='rect',
                        **{'width': "2",
                           'tooltip': "TODO",
                           'href': "TODO",
                           'target': "_blank"})

            with ws.subgraph(name='cluster_ws_dashboards') as wsd:
                wsd.body.append(r'label="Dashboard"')
                wsd.body.append('style="filled"')
                wsd.body.append('color="#808080"')
                wsd.body.append('fillcolor="#e4f2f7"')

                wsd.node("dashboard","Dashboard", fillcolor='#87CEEB', style='filled', shape='rect',
                        **{'width': "2",
                           'tooltip': "TODO",
                           'href': "TODO",
                           'target': "_blank"})

    dot.edge('blank1','eval_ds', style='invis')
    dot.edge('notebook_model','blank1', style='invis')
    dot.edge('eval_ds','registered_model', style='invis')
    dot.edge('eval_ds','eval_run', arrowhead='normal')
    dot.edge('notebook_model', 'eval_run', lhead='cluster_ws_experiment_model', arrowhead='normal')
    dot.edge('eval_run', 'registered_model', ltail='cluster_ws_experiment_model', arrowhead='normal')
    dot.edge('eval_run', 'feedback_agent', lhead='cluster_ws_end_point', style='invis')
    dot.edge('registered_model', 'served_model', lhead='cluster_ws_end_point', arrowhead='normal')
    dot.edge('registered_model','inference', style='invis')
    dot.edge('served_model', 'inference',arrowhead='normal')
    dot.edge('inference', 'blank2', style='invis')
    dot.edge('served_model','metrics_nb', style='invis')
    dot.edge('inference', 'metrics_nb', arrowhead='normal')
    dot.edge('metrics_nb', 'dashboard', arrowhead='normal')

    html = dot._repr_image_svg_xml().format(
        agent_deployment_label=f'{config.agent_name} Agent Deployment Entity Diagram',
        unity_catalog_label=f'Unity Catalog: <a href="{config.schema.agents.uc_relative_url}" target="_blank">{config.schema.agents.uc_name}</a>',
        experiment_label=f'Experiment: <a href="{config.experiment.ws_relative_url}" target="_blank">{config.experiment.experiment_name.split("/")[-1]}</a>',
        end_point="End Point",
        model_run=f'<a href="{config.experiment.ws_exp_release_relative_url}" target="_blank">{config.experiment.model_run_name}</a>')

    html = re.sub(r'<svg width=\"\d*pt\" height=\"\d*pt\"',
                  '<div style="text-align:center;"><svg width="800pt" aligned=center', html)

    html = re.sub(r'stroke-width=\"2\"', 'stroke-width=\"4\"', html)

    return html