#!/usr/bin/env python3

"""
This script wraps all CSS selectors in style.css with .fill-in-the-blank-experiment
to scope the styles to only that experiment page.
"""

import re
import sys

def scope_css(css_content, scope_class):
    """
    Scope all CSS selectors with a wrapper class.
    """
    lines = []
    result = []
    in_comment = False
    in_media_query = False
    media_query_depth = 0
    
    # Split into lines for easier processing
    for line in css_content.split('\n'):
        # Track multi-line comments
        if '/*' in line:
            in_comment = True
        if '*/' in line:
            in_comment = False
            result.append(line)
            continue
            
        if in_comment:
            result.append(line)
            continue
        
        # Track @media queries and other @-rules
        if re.match(r'\s*@', line):
            result.append(line)
            if '{' in line:
                in_media_query = True
                media_query_depth += line.count('{')
                media_query_depth -= line.count('}')
            continue
        
        if in_media_query:
            media_query_depth += line.count('{')
            media_query_depth -= line.count('}')
            if media_query_depth == 0:
                in_media_query = False
        
        # Check if this line contains a selector (ends with { or is followed by {)
        stripped = line.strip()
        
        # Skip empty lines and lines with only {}
        if not stripped or stripped in ['{', '}']:
            result.append(line)
            continue
        
        # If line contains a selector (before opening brace)
        if '{' in line and not line.strip().startswith('}'):
            # Split selector from rules
            parts = line.split('{', 1)
            selector = parts[0].strip()
            rest = parts[1] if len(parts) > 1 else ''
            
            # Scope the selector
            scoped_selector = scope_selector(selector, scope_class)
            
            # Reconstruct the line
            indent = len(line) - len(line.lstrip())
            result.append(' ' * indent + scoped_selector + '{' + rest)
        else:
            # Not a selector line (might be properties, closing braces, etc.)
            result.append(line)
    
    return '\n'.join(result)

def scope_selector(selector, scope_class):
    """
    Scope a single selector or comma-separated selectors.
    """
    # Handle comma-separated selectors
    if ',' in selector:
        selectors = [s.strip() for s in selector.split(',')]
        scoped = [scope_single_selector(s, scope_class) for s in selectors]
        return ', '.join(scoped)
    
    return scope_single_selector(selector, scope_class)

def scope_single_selector(selector, scope_class):
    """
    Scope a single selector.
    """
    selector = selector.strip()
    
    # Don't scope if already scoped
    if selector.startswith(scope_class):
        return selector
    
    # Replace html with scope class
    if selector == 'html' or selector.startswith('html ') or selector.startswith('html:') or selector.startswith('html.'):
        return re.sub(r'^html\b', scope_class, selector)
    
    # Replace body with scope class  
    if selector == 'body' or selector.startswith('body ') or selector.startswith('body:') or selector.startswith('body.'):
        return re.sub(r'^body\b', scope_class, selector)
    
    # For all other selectors, prepend the scope class
    return f'{scope_class} {selector}'

if __name__ == '__main__':
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'style.css')
    output_file = os.path.join(script_dir, 'style.scoped.css')
    scope_class = '.fill-in-the-blank-experiment'
    
    # Read the input file
    with open(input_file, 'r') as f:
        css = f.read()
    
    # Scope the CSS
    scoped_css = scope_css(css, scope_class)
    
    # Write the output file
    with open(output_file, 'w') as f:
        f.write(scoped_css)
    
    print(f'✓ Created scoped CSS file: {output_file}')
    print(f'  Original size: {len(css)} bytes')
    print(f'  Scoped size: {len(scoped_css)} bytes')
