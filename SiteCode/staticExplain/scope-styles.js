#!/usr/bin/env node

/**
 * This script wraps all CSS selectors in style.css with .fill-in-the-blank-experiment
 * to scope the styles to only that experiment page.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const inputFile = path.join(__dirname, 'style.css');
const outputFile = path.join(__dirname, 'style.scoped.css');

// Read the original CSS
const css = fs.readFileSync(inputFile, 'utf8');

// Function to scope a CSS selector
function scopeSelector(selector, scopeClass) {
  // Trim the selector
  selector = selector.trim();
  
  // Don't scope @-rules (like @media, @keyframes, etc.)
  if (selector.startsWith('@')) {
    return selector;
  }
  
  // Handle multiple selectors (comma-separated)
  if (selector.includes(',')) {
    return selector
      .split(',')
      .map(s => scopeSelector(s.trim(), scopeClass))
      .join(', ');
  }
  
  // Replace html with the scope class
  if (selector === 'html' || selector.startsWith('html ') || selector.startsWith('html:') || selector.startsWith('html.')) {
    selector = selector.replace(/^html/, scopeClass);
  }
  // Replace body with the scope class
  else if (selector === 'body' || selector.startsWith('body ') || selector.startsWith('body:') || selector.startsWith('body.')) {
    selector = selector.replace(/^body/, scopeClass);
  }
  // For all other selectors, prepend the scope class
  else if (!selector.startsWith(scopeClass)) {
    selector = `${scopeClass} ${selector}`;
  }
  
  return selector;
}

// Process the CSS
function scopeCSS(css, scopeClass) {
  let output = '';
  let inComment = false;
  let inAtRule = false;
  let atRuleDepth = 0;
  let buffer = '';
  let selectorBuffer = '';
  
  for (let i = 0; i < css.length; i++) {
    const char = css[i];
    const nextChar = css[i + 1];
    
    // Handle comments
    if (char === '/' && nextChar === '*') {
      inComment = true;
      output += char;
      continue;
    }
    if (inComment) {
      output += char;
      if (char === '*' && nextChar === '/') {
        inComment = false;
        i++;
        output += '/';
      }
      continue;
    }
    
    // Track @-rules (media queries, keyframes, etc.)
    if (char === '@') {
      inAtRule = true;
      selectorBuffer = '';
    }
    
    // Track curly braces for @-rules
    if (inAtRule && char === '{') {
      atRuleDepth++;
      output += selectorBuffer + char;
      selectorBuffer = '';
      continue;
    }
    
    if (inAtRule && char === '}') {
      atRuleDepth--;
      if (atRuleDepth === 0) {
        inAtRule = false;
      }
      output += char;
      continue;
    }
    
    // If we're inside an @-rule block, just pass through
    if (inAtRule && atRuleDepth > 0) {
      // Need to scope selectors inside @media, etc.
      if (char === '{' && selectorBuffer.trim()) {
        const scopedSelector = scopeSelector(selectorBuffer, scopeClass);
        output += scopedSelector + char;
        selectorBuffer = '';
      } else if (char === '}') {
        output += char;
        selectorBuffer = '';
      } else {
        selectorBuffer += char;
      }
      continue;
    }
    
    if (inAtRule && atRuleDepth === 0) {
      selectorBuffer += char;
      continue;
    }
    
    // Normal CSS rules
    if (char === '{') {
      const scopedSelector = scopeSelector(selectorBuffer, scopeClass);
      output += scopedSelector + char;
      selectorBuffer = '';
    } else if (char === '}') {
      output += char;
      selectorBuffer = '';
    } else {
      selectorBuffer += char;
    }
  }
  
  return output;
}

const scopedCSS = scopeCSS(css, '.fill-in-the-blank-experiment');

// Write the scoped CSS
fs.writeFileSync(outputFile, scopedCSS, 'utf8');

console.log(`✓ Created scoped CSS file: ${outputFile}`);
console.log(`  Original size: ${css.length} bytes`);
console.log(`  Scoped size: ${scopedCSS.length} bytes`);
