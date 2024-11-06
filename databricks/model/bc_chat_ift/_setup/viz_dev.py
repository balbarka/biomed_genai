# Databricks notebook source
# MAGIC %run ./setup_bc_chat_ift

# COMMAND ----------

# MAGIC %r
# MAGIC def agent_deploy_graphic(config):
# MAGIC     import re
# MAGIC     from graphviz import Digraph
# MAGIC
# MAGIC     dot = Digraph('pt')
# MAGIC     dot.attr(compound='true')
# MAGIC     dot.graph_attr['rankdir'] = 'LR'
# MAGIC     dot.edge_attr.update(arrowhead='none', arrowsize='1')
# MAGIC     dot.attr('node', shape='rectangle')
# MAGIC
# MAGIC     with dot.subgraph(name='cluster_agent_deployment') as ad:
# MAGIC         ad.body.append(r'label="{agent_deployment_label}"')
# MAGIC
# MAGIC         with ad.subgraph(name='cluster_uc') as uc:
# MAGIC             uc.body.append(r'label="{unity_catalog_label}"')
# MAGIC             uc.body.append('style="filled"')
# MAGIC             uc.body.append('color="#808080"')
# MAGIC             uc.body.append('fillcolor="#F5F5F5"')
# MAGIC
# MAGIC             uc.node('blank1', 'blank1', shape='rect', style='invis', **{'width': "2"})
# MAGIC
# MAGIC             with uc.subgraph(name='cluster_uc_delta') as ucd:
# MAGIC                 ucd.body.append(r'label="Delta (DataSet)"')
# MAGIC                 ucd.body.append('style="filled"')
# MAGIC                 ucd.body.append('color="#808080"')
# MAGIC                 ucd.body.append('fillcolor="#e4f2f7"')
# MAGIC
# MAGIC                 ucd.node('eval_ds', config.experiment.eval_ds.table, fillcolor='#CAD9EF', style='filled', shape='rect',
# MAGIC                     **{'width': "2", 'tooltip': "Evaluation Dataset",
# MAGIC                        'href': config.experiment.eval_ds.uc_relative_url, 'target': "_blank"})
# MAGIC
# MAGIC             with uc.subgraph(name='cluster_uc_model') as ucm:
# MAGIC                 ucm.body.append(r'label="Registered Model"')
# MAGIC                 ucm.body.append('style="filled"')
# MAGIC                 ucm.body.append('color="#808080"')
# MAGIC                 ucm.body.append('fillcolor="#e4f2f7"')
# MAGIC
# MAGIC                 ucm.node("registered_model", f'{config.registered_model.model_name} (v{config.release_version})',
# MAGIC                          fillcolor='#87CEEB', style='filled', shape='rect',
# MAGIC                          **{'width': "2", 'tooltip': "Registered Model",
# MAGIC                             'href': config.registered_model.uc_relative_url, 'target': "_blank"})
# MAGIC                 
# MAGIC             with uc.subgraph(name='cluster_uc_inference') as uci:
# MAGIC                 uci.body.append(r'label="Delta (Inference)"')
# MAGIC                 uci.body.append('style="filled"')
# MAGIC                 uci.body.append('color="#808080"')
# MAGIC                 uci.body.append('fillcolor="#e4f2f7"')
# MAGIC
# MAGIC                 uci.node("inference", "Inference Table",
# MAGIC                          fillcolor='#87CEEB', style='filled', shape='rect',
# MAGIC                          **{'width': "2", 'tooltip': "Inference Table",
# MAGIC                             'href': config.registered_model.uc_relative_url, 'target': "_blank"})
# MAGIC
# MAGIC             uc.node('blank2', 'blank2', shape='rect', style='invis', **{'width': "2"})
# MAGIC
# MAGIC         with ad.subgraph(name='cluster_ws') as ws:
# MAGIC             ws.body.append('label="Workspace"')
# MAGIC             ws.body.append('labelloc="b"')
# MAGIC             ws.body.append('style="filled"')
# MAGIC             ws.body.append('color="#808080"')
# MAGIC             ws.body.append('fillcolor="#F5F5F5"')
# MAGIC             
# MAGIC             with ws.subgraph(name='cluster_nb') as nb:
# MAGIC                 nb.body.append('label="Notebook (Model)"')
# MAGIC                 nb.body.append('labelloc="b"')
# MAGIC                 nb.body.append('style="filled"')
# MAGIC                 nb.body.append('color="#808080"')
# MAGIC                 nb.body.append('fillcolor="#e4f2f7"')
# MAGIC
# MAGIC                 nb.node('notebook_model', config.default_model_name or 'MODEL',
# MAGIC                     fillcolor='#87CEEB', style='filled', shape='rect',
# MAGIC                     **{'width': "1.78", 'tooltip': "Notebook Model"})
# MAGIC
# MAGIC             with ws.subgraph(name='cluster_ws_experiment') as exp:
# MAGIC                 exp.body.append(r'label="{experiment_label}"')
# MAGIC                 exp.body.append('style="filled"')
# MAGIC                 exp.body.append('color="#808080"')
# MAGIC                 exp.body.append('fillcolor="#e4f2f7"')
# MAGIC
# MAGIC                 with exp.subgraph(name='cluster_ws_experiment_model') as mdl:
# MAGIC                     mdl.body.append(r'label="{model_run}"')
# MAGIC                     mdl.body.append('style="filled"')
# MAGIC                     mdl.body.append('color="#808080"')
# MAGIC                     mdl.body.append('fillcolor="#87CEEB"')
# MAGIC                 
# MAGIC                     mdl.node("eval_run", config.experiment.eval_ds.ds_release_version_name,
# MAGIC                              fillcolor='#CAD9EF', style='filled', shape='rect',
# MAGIC                              **{'width': "2", 'tooltip': "Evaluation Dataset",
# MAGIC                             'href': config.experiment.ws_exp_release_relative_url, 'target': "_blank"})
# MAGIC
# MAGIC             with ws.subgraph(name='cluster_ws_end_point') as ep:
# MAGIC                 ep.body.append(r'label="{end_point}"')
# MAGIC                 ep.body.append('style="filled"')
# MAGIC                 ep.body.append('color="#808080"')
# MAGIC                 ep.body.append('fillcolor="#e4f2f7"')
# MAGIC
# MAGIC                 ep.node("served_model","Agent Model", fillcolor='#87CEEB', style='filled', shape='rect',
# MAGIC                         **{'width': "2",
# MAGIC                            'tooltip': "Serving Entity",
# MAGIC                            'href': "https://adb-830292400663869.9.azuredatabricks.net/ml/endpoints/agents_biomed_genai-agents-bc_qa_chat?o=830292400663869",
# MAGIC                            'target': "_blank"})
# MAGIC
# MAGIC                 ep.node("feedback_agent","Review App", fillcolor='#87CEEB', style='filled', shape='rect',
# MAGIC                         **{'width': "2",
# MAGIC                            'tooltip': "Review App",
# MAGIC                            'href': "https://adb-830292400663869.9.azuredatabricks.net/ml/review/biomed_genai.agents.bc_qa_chat/1/chat?o=830292400663869",
# MAGIC                            'target': "_blank"})
# MAGIC
# MAGIC             with ws.subgraph(name='cluster_ws_metrics') as wsm:
# MAGIC                 wsm.body.append(r'label="Notebook (Metrics)"')
# MAGIC                 wsm.body.append('style="filled"')
# MAGIC                 wsm.body.append('color="#808080"')
# MAGIC                 wsm.body.append('fillcolor="#e4f2f7"')
# MAGIC
# MAGIC                 wsm.node("metrics_nb","bc_qa_chat_metrics", fillcolor='#87CEEB', style='filled', shape='rect',
# MAGIC                         **{'width': "2",
# MAGIC                            'tooltip': "TODO",
# MAGIC                            'href': "TODO",
# MAGIC                            'target': "_blank"})
# MAGIC
# MAGIC             with ws.subgraph(name='cluster_ws_dashboards') as wsd:
# MAGIC                 wsd.body.append(r'label="Dashboard"')
# MAGIC                 wsd.body.append('style="filled"')
# MAGIC                 wsd.body.append('color="#808080"')
# MAGIC                 wsd.body.append('fillcolor="#e4f2f7"')
# MAGIC
# MAGIC                 wsd.node("dashboard","Dashboard", fillcolor='#87CEEB', style='filled', shape='rect',
# MAGIC                         **{'width': "2",
# MAGIC                            'tooltip': "TODO",
# MAGIC                            'href': "TODO",
# MAGIC                            'target': "_blank"})
# MAGIC
# MAGIC     dot.edge('blank1','eval_ds', style='invis')
# MAGIC     dot.edge('notebook_model','blank1', style='invis')
# MAGIC     dot.edge('eval_ds','registered_model', style='invis')
# MAGIC     dot.edge('eval_ds','eval_run', arrowhead='normal')
# MAGIC     dot.edge('notebook_model', 'eval_run', lhead='cluster_ws_experiment_model', arrowhead='normal')
# MAGIC     dot.edge('eval_run', 'registered_model', ltail='cluster_ws_experiment_model', arrowhead='normal')
# MAGIC     dot.edge('eval_run', 'feedback_agent', lhead='cluster_ws_end_point', style='invis')
# MAGIC     dot.edge('registered_model', 'served_model', lhead='cluster_ws_end_point', arrowhead='normal')
# MAGIC     dot.edge('registered_model','inference', style='invis')
# MAGIC     dot.edge('served_model', 'inference',arrowhead='normal')
# MAGIC     dot.edge('inference', 'blank2', style='invis')
# MAGIC     dot.edge('served_model','metrics_nb', style='invis')
# MAGIC     dot.edge('inference', 'metrics_nb', arrowhead='normal')
# MAGIC     dot.edge('metrics_nb', 'dashboard', arrowhead='normal')
# MAGIC
# MAGIC     html = dot._repr_image_svg_xml().format(
# MAGIC         agent_deployment_label=f'{config.agent_name} Agent Deployment Entity Diagram',
# MAGIC         unity_catalog_label=f'Unity Catalog: <a href="{config.schema.agents.uc_relative_url}" target="_blank">{config.schema.agents.uc_name}</a>',
# MAGIC         experiment_label=f'Experiment: <a href="{config.experiment.ws_relative_url}" target="_blank">{config.experiment.experiment_name.split("/")[-1]}</a>',
# MAGIC         end_point="End Point",
# MAGIC         model_run=f'<a href="{config.experiment.ws_exp_release_relative_url}" target="_blank">{config.experiment.model_run_name}</a>')
# MAGIC
# MAGIC     html = re.sub(r'<svg width=\"\d*pt\" height=\"\d*pt\"',
# MAGIC                   '<div style="text-align:center;"><svg width="800pt" aligned=center', html)
# MAGIC
# MAGIC     html = re.sub(r'stroke-width=\"2\"', 'stroke-width=\"4\"', html)
# MAGIC
# MAGIC     return html
