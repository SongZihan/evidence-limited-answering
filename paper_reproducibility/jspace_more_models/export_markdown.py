"""Export the verified comprehensive HTML report to portable Markdown."""
from html.parser import HTMLParser
from pathlib import Path
import re


class Node:
    def __init__(self, tag='', attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []


class Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in {'meta', 'br', 'hr', 'img', 'link', 'input'}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def descendants(node, tag):
    if isinstance(node, str):
        return
    if node.tag == tag:
        yield node
    for child in node.children:
        yield from descendants(child, tag)


def render(node):
    if isinstance(node, str):
        return node
    tag = node.tag
    if tag in {'head', 'style', 'script', 'nav'}:
        return ''
    if tag == 'table':
        rows = []
        for row in descendants(node, 'tr'):
            cells = [render(c).strip().replace('|', '\\|').replace('\n', '<br>')
                     for c in row.children if isinstance(c, Node) and c.tag in {'td', 'th'}]
            rows.append('| ' + ' | '.join(cells) + ' |')
        if not rows:
            return ''
        width = sum(isinstance(c, Node) and c.tag in {'td', 'th'}
                    for c in next(descendants(node, 'tr')).children)
        rows.insert(1, '| ' + ' | '.join(['---'] * width) + ' |')
        return '\n\n' + '\n'.join(rows) + '\n\n'
    text = ''.join(render(c) for c in node.children)
    if tag in {'h1', 'h2', 'h3', 'h4'}:
        return '\n\n' + '#' * int(tag[1]) + ' ' + text.strip().replace('\n', '：') + '\n\n'
    if tag in {'p', 'section', 'header', 'ul', 'ol', 'div'}:
        return '\n\n' + text.strip() + '\n\n'
    if tag in {'b', 'strong'}:
        return '**' + text + '**'
    if tag in {'em', 'i'}:
        return '*' + text + '*'
    if tag == 'code':
        return '`' + text + '`'
    if tag == 'a':
        return '[' + text + '](' + node.attrs.get('href', '') + ')'
    if tag == 'blockquote':
        return '\n\n' + '\n'.join('> ' + line for line in text.strip().splitlines()) + '\n\n'
    if tag == 'li':
        return '\n- ' + text.strip() + '\n'
    if tag == 'br':
        return '\n'
    return text


base = Path(__file__).resolve().parent / 'results'
parser = Parser()
parser.feed((base / 'comprehensive_report.html').read_text(encoding='utf-8'))
markdown = re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', render(parser.root)).strip() + '\n'
output = base / '联合抽取关系归属实验_整体结果与说明_2026-10-15.md'
output.write_text(markdown, encoding='utf-8')
assert markdown.count('\n## ') == 12
assert sum(line.startswith('| ---') for line in markdown.splitlines()) == len(list(descendants(parser.root, 'table')))
assert '24612' in markdown and '1811/3516' in markdown
assert '<style>' not in markdown
links = re.findall(r'\]\(([^)]+)\)', markdown)
for link in links:
    if not link.startswith(('http:', 'https:', '#')):
        assert (base / link).exists(), link
print(f'Created: {output}')
print(f'Sections: 12; tables: {len(list(descendants(parser.root, "table")))}; verified links: {len(links)}')
