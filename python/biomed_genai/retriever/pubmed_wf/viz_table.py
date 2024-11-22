"""Vizualizations has methods that return html based upon biomed configurations
This is helpful to provide a graphical representation of the workflow while also refrencing the represented UC entities.
NOTE: workflow_graphic requires the installation of graphviz"""

def workflow_table(config):
    inspect_html = f"""<table border="1" cellpadding="10">
    <tr><th style="background-color: orange;">BioMed Asset</th>
        <th style="background-color: orange;">Attributes</th>
        <th style="background-color: orange;">Description</th></tr>
    <tr><td ROWSPAN=3 style="background-color: yellow;"><b>raw_metadata_xml</b></td>
        <td><b>ddl</b>: <a href={config.raw_metadata_xml.sql_relative_url} style="text-decoration:none">{config.raw_metadata_xml.sql_file}</a></td>
        <td ROWSPAN=3><b>raw_metadata_xml</b> is the table that syncs with all PubMed articles list.</br>It will also maintain the download status of all articles.</td></tr>
    <tr><td><b>table</b>: <a href={config.raw_metadata_xml.uc_relative_url} style="text-decoration:none">{config.raw_metadata_xml.name}</a></td></tr>
    <tr><td><b>cp</b>: <a href={config.raw_metadata_xml.cp.uc_relative_url} style="text-decoration:none">{config.raw_metadata_xml.cp.name}</a></td></tr>
    <tr><td ROWSPAN=2 style="background-color: yellow;"><b>raw_search_hist</b></td>
        <td><b>ddl</b>: <a href={config.raw_search_hist.sql_relative_url} style="text-decoration:none">{config.raw_search_hist.sql_file}</a></td>
        <td ROWSPAN=2><b>raw_search_hist</b> Is where we will store previous searches used to avoid having larger search windows.</td></tr>
    <tr><td><b>table</b>: <a href={config.raw_search_hist.uc_relative_url} style="text-decoration:none">{config.raw_search_hist.name}</a></td></tr>
    <tr><td ROWSPAN=2 style="background-color: yellow;"><b>raw_articles_xml</b></td>
        <td><b>ddl</b>: <a href={config.raw_articles_xml.sql_relative_url} style="text-decoration:none">{config.raw_articles_xml.sql_file}</a></td>
        <td ROWSPAN=2><b>raw_articles_xml</b> is the volume that contains articles as xml files.</br>It will also maintain the download status of all articles.</td></tr>
    <tr><td><b>data</b>: <a href={config.raw_articles_xml.uc_relative_url} style="text-decoration:none">{config.raw_articles_xml.name}</a></td></tr>
    <tr><td ROWSPAN=3 style="background-color: yellow;"><b>curated_articles</b></td>
        <td><b>ddl</b>: <a href={config.curated_articles_xml.sql_relative_url} style="text-decoration:none">{config.curated_articles_xml.sql_file}</a></td>
        <td ROWSPAN=3><b>curated_articles</b> is the table that contains a high performance format Delta for evaluating the XML Downloads</td></tr>
    <tr><td><b>table</b>: <a href={config.curated_articles_xml.uc_relative_url} style="text-decoration:none">{config.curated_articles_xml.name}</a></td></tr>
    <tr><td><b>cp</b>: <a href={config.curated_articles_xml.cp.uc_relative_url} style="text-decoration:none">{config.curated_articles_xml.cp.name}</a></td></tr>
    <tr><td ROWSPAN=3 style="background-color: yellow;"><b>processed_articles_content</b></td>
        <td><b>ddl</b>: <a href={config.processed_articles_content.sql_relative_url} style="text-decoration:none">{config.processed_articles_content.sql_file}</a></td>
        <td ROWSPAN=3><b>raw_metadata</b> is the table that syncs with all PubMed articles list.</br>It will also maintain the download status of all articles.</td></tr>
    <tr><td><b>table</b>: <a href={config.processed_articles_content.uc_relative_url} style="text-decoration:none">{config.processed_articles_content.name}</a></td></tr>
    <tr><td><b>cp</b>: <a href={config.processed_articles_content.cp.uc_relative_url} style="text-decoration:none">{config.processed_articles_content.cp.name}</a></td></tr> 
    
    </table>"""
    return inspect_html

