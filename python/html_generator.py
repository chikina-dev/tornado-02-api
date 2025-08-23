import markdown
from typing import List, Dict, Any

def generate_analysis_html(analysis_result: str) -> str:
    """
    Generates an HTML report for the user profile analysis.
    """
    css_style = '''
    <style>
        body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 20px auto; padding: 0 20px; background-color: #f9f9f9; }
        h1 { color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .analysis-content { background-color: #ffffff; border: 1px solid #ecf0f1; border-radius: 8px; padding: 25px; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        h2 { color: #3498db; margin-top: 0; }
        p { margin-bottom: 15px; }
        strong { color: #2980b9; }
    </style>
    '''
    
    html_content = markdown.markdown(analysis_result)

    html_output = f'''
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ユーザープロファイル分析結果</title>
        {css_style}
    </head>
    <body>
        <h1>ユーザープロファイル分析結果</h1>
        <div class="analysis-content">
            {html_content}
        </div>
    </body>
    </html>
    '''
    return html_output
