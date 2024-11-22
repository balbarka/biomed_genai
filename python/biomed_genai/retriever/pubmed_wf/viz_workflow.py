"""
Curation Workflow HTML for all schemas in price transparency
"""

def workflow_graphic(config):
    import re
    from graphviz import Digraph

    dot = Digraph('pt')
    dot.attr(compound='true')
    dot.graph_attr['rankdir'] = 'LR'
    dot.edge_attr.update(arrowhead='none', arrowsize='2')
    dot.attr('node', shape='rectangle')

    def uc_link(url, ttip=''):
        return {'tooltip': ttip, 'href': url, 'target': "_blank",
                'width': "1.5"}
        
    def uc_table_node(cluster, name, hyperlink=''):
        cluster.node(name, name,
                    fillcolor='#CAD9EF', style='filled', shape='rect',
                    **{'width': "2",
                       'tooltip': name,
                       'href': hyperlink,
                       'target': "_blank"})

    def uc_ds_node(cluster, name, hyperlink=''):
        cluster.node(name, name,
                    fillcolor='#CAD9EF', style='filled', shape='note',
                    **{'width': "2",
                       'tooltip': name,
                       'href': hyperlink,
                       'target': "_blank"})


    def uc_folder_node(cluster, name, hyperlink=''):
        cluster.node("fold_"+name, name,
                    fillcolor='#ffffbb', style='filled', shape='folder',
                    **{'width': "2",
                       'tooltip': name,
                       'href': hyperlink,
                       'target': "_blank"})

    def pmc_node(cluster, name, hyperlink='https://pubmed.ncbi.nlm.nih.gov/'):
       cluster.node(name, name,
                    fillcolor='#e4f2f7', style='filled', shape='component',
                    **{'width': "2",
                       'tooltip': name,
                       'href': hyperlink,
                       'target': "_blank"})

    def index_node(cluster, name, hyperlink=''):
       cluster.node(name, name,
                    fillcolor='#C1E1C1', style='filled', shape='tab',
                    **{'tooltip': name,
                       'href': hyperlink,
                       'target': "_blank"})

    with dot.subgraph(name='cluster_biomed_workflow') as pp:
        pp.body.append('label="{BioMed Workflow}"')
        with pp.subgraph(name='cluster_pmc') as pmc:
            pmc.body.append('label="{PMC}"')
            pmc.body.append('style="filled"')
            pmc.body.append('color="#808080"')
            pmc.body.append('fillcolor="#F5F5F5"')

            pmc_node(pmc, "pmc_metadata", hyperlink="https://www.ncbi.nlm.nih.gov/pmc/tools/get-metadata/")
            pmc_node(pmc, "pmc_articles", hyperlink="https://www.ncbi.nlm.nih.gov/guide/howto/obtain-full-text/")
            pmc_node(pmc, "pmc_search", hyperlink="https://pubmed.ncbi.nlm.nih.gov/help/")  

        with pp.subgraph(name='cluster_raw') as raw:
            raw.body.append('label="{Raw Schema}"')
            raw.body.append('style="filled"')
            raw.body.append('color="#808080"')
            raw.body.append('fillcolor="#F5F5F5"')
            
            uc_table_node(raw, 'metadata_xml', hyperlink=config.raw_metadata_xml.uc_relative_url)
            uc_table_node(raw, 'search_hist', hyperlink=config.raw_search_hist.uc_relative_url)
            raw.node('raw_node_xml', '', fillcolor='red', shape='point')
            uc_folder_node(raw, 'articles_xml', hyperlink=config.raw_articles_xml.uc_relative_url)

        with pp.subgraph(name='cluster_curated') as cur:
            cur.body.append('label="{Curated Schema}"')
            cur.body.append('style="filled"')
            cur.body.append('color="#808080"')
            cur.body.append('fillcolor="#F5F5F5"')

            uc_table_node(cur, 'articles_xml', hyperlink=config.curated_articles_xml.uc_relative_url)        

        with pp.subgraph(name='cluster_processed') as pro:
            pro.body.append('label="{Processed Schema}"')
            pro.body.append('style="filled"')
            pro.body.append('color="#808080"')
            pro.body.append('fillcolor="#F5F5F5"')

            uc_table_node(pro, 'articles_content', hyperlink=config.processed_articles_content.uc_relative_url)
            index_node(pro, 'articles_content_vs_index', hyperlink=config.vector_search.biomed.processed_articles_content_vs_index.ws_relative_url)

        dot.edge('pmc_search', 'search_hist')
        dot.edge('pmc_articles', 'raw_node_xml')
        dot.edge('pmc_metadata', 'metadata_xml')
        
        dot.edge('raw_node_xml', 'metadata_xml')
        dot.edge('raw_node_xml', 'search_hist')

        dot.edge('metadata_xml', 'articles_xml', style='invis')
        dot.edge('search_hist', 'fold_articles_xml', style='invis')
        
        dot.edge('raw_node_xml', 'fold_articles_xml')

        dot.edge('fold_articles_xml', 'articles_xml')

        dot.edge('articles_xml', 'articles_content')
        
        dot.edge('articles_content', 'articles_content_vs_index')        

    html = dot._repr_image_svg_xml()

    html = re.sub(r'<svg width=\"\d*pt\" height=\"\d*pt\"',
                  '<div style="text-align:center;"><svg width="800pt" aligned=center', html)
    html = re.sub(r'{BioMed Workflow}',
                  f'<a href="https://github.com/balbarka/biomed_genai" target="_blank">pubmed_wf Workflow</a>',
                  html)
    html = re.sub(r'{PMC}',
                  f'<a href="https://pubmed.ncbi.nlm.nih.gov/" target="_blank">PMC</a>',
                  html)
    html = re.sub(r'{Raw Schema}',
                  f'<a href="{config.schema.raw.uc_relative_url}" target="_blank">{config.schema.raw.name}</a>',
                  html)
    html = re.sub(r'{Curated Schema}',
                  f'<a href="{config.schema.curated.uc_relative_url}" target="_blank">{config.schema.curated.name}</a>',
                  html)
    html = re.sub(r'{Processed Schema}',
                  f'<a href="{config.schema.processed.uc_relative_url}" target="_blank">{config.schema.processed.name}</a>',
                  html)

    html = re.sub(r'stroke-width=\"2\"', 'stroke-width=\"4\"', html)

    return html