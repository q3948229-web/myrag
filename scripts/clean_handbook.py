import re
import json

def clean_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    current_buffer = ""

    # Patterns to detect headings that might not be marked with #
    chapter_p = re.compile(r'^第[一二三四五六七八九十百]+章\s*(.*)')
    section_p = re.compile(r'^第[一二三四五六七八九十百]+节\s*(.*)')
    article_p = re.compile(r'^第[一二三四五六七八九十百]+条\s*(.*)')
    
    # TOC/Page number pattern
    toc_p = re.compile(r'\.{3,}\s*\d+')

    for line in lines:
        line = line.strip()
        if not line:
            if current_buffer:
                cleaned_lines.append(current_buffer)
                current_buffer = ""
            continue

        # Skip TOC lines with page numbers
        if toc_p.search(line):
            continue

        # Check for headings
        if line.startswith('##'):
            if current_buffer:
                cleaned_lines.append(current_buffer)
                current_buffer = ""
            cleaned_lines.append(line)
            continue
        
        # Merge logic: if line doesn't end with sentence-ending punctuation 
        # and next line seems like continuation
        if current_buffer:
            # If current buffer ends with punctuation, start new
            if current_buffer[-1] in '。！？】：”"':
                cleaned_lines.append(current_buffer)
                current_buffer = line
            else:
                # Merge with a space if it's alphanumeric, otherwise just merge
                if re.match(r'^[a-zA-Z0-9]', line):
                    current_buffer += " " + line
                else:
                    current_buffer += line
        else:
            current_buffer = line

    if current_buffer:
        cleaned_lines.append(current_buffer)

    return cleaned_lines

def parse_to_nodes(cleaned_lines):
    nodes = []
    
    current_id = 1
    
    # Hierarchy state
    current_doc = "上海大学本科生学生手册"
    current_chapter = None
    current_section = None
    
    stack = [] # (level, id)

    for line in cleaned_lines:
        node_type = 'content'
        title = ""
        level = 0
        number = ""
        content = ""
        
        # Determine node type
        if line.startswith('##'):
            text = line.replace('##', '').strip()
            title = text
            if "章" in text and text.startswith('第'):
                node_type = 'chapter'
                level = 1
            elif "节" in text and text.startswith('第'):
                node_type = 'section'
                level = 2
            else:
                node_type = 'heading'
                level = 1
        elif line.startswith('第') and '条' in line[:10]:
            node_type = 'article'
            level = 3
            match = re.search(r'^(第[一二三四五六七八九十百]+条)\s*(.*)', line)
            if match:
                number = match.group(1)
                content = match.group(2)
            else:
                content = line
        else:
            node_type = 'text'
            level = 4
            content = line

        # Assign unique ID and Parent
        # This is a simplified tree builder
        parent_id = None
        # Find parent based on level
        while stack and stack[-1][0] >= level and level != 0:
            stack.pop()
        
        if stack:
            parent_id = stack[-1][1]
        
        nodes.append({
            'id': current_id,
            'parent_id': parent_id,
            'level': level,
            'node_type': node_type,
            'title': title,
            'number': number,
            'content': content,
            'path': "" # To be filled
        })
        
        if level > 0:
            stack.append((level, current_id))
            
        current_id += 1

    # Fill paths
    id_to_node = {n['id']: n for n in nodes}
    for n in nodes:
        path = []
        curr = n
        while curr:
            segment = curr['title'] or curr['number'] or (curr['content'][:20] + "...")
            path.append(segment)
            curr = id_to_node.get(curr['parent_id'])
        n['path'] = "/".join(reversed(path))

    return nodes

def generate_sql(nodes):
    sql_lines = [
        "CREATE TABLE IF NOT EXISTS handbook_nodes (",
        "    id INTEGER PRIMARY KEY,",
        "    parent_id INTEGER,",
        "    level INTEGER NOT NULL,",
        "    node_type VARCHAR(50) NOT NULL,",
        "    title TEXT,",
        "    number VARCHAR(50),",
        "    content TEXT,",
        "    path TEXT,",
        "    FOREIGN KEY (parent_id) REFERENCES handbook_nodes(id)",
        ");",
        "CREATE INDEX IF NOT EXISTS idx_handbook_path ON handbook_nodes(path);"
    ]
    
    for n in nodes:
        # Escape single quotes
        title = n['title'].replace("'", "''")
        number = n['number'].replace("'", "''")
        content = n['content'].replace("'", "''")
        path = n['path'].replace("'", "''")
        parent_id = n['parent_id'] if n['parent_id'] is not None else "NULL"
        
        sql = f"INSERT INTO handbook_nodes (id, parent_id, level, node_type, title, number, content, path) VALUES ({n['id']}, {parent_id}, {n['level']}, '{n['node_type']}', '{title}', '{number}', '{content}', '{path}');"
        sql_lines.append(sql)
        
    return "\n".join(sql_lines)

if __name__ == "__main__":
    md_path = "d:/myrag/data/raw/2025年本科生学生手册.md"
    cleaned = clean_markdown(md_path)
    
    # Save cleaned MD
    with open("d:/myrag/data/processed/2025年本科生学生手册_cleaned.md", "w", encoding="utf-8") as f:
        f.write("\n\n".join(cleaned))
        
    nodes = parse_to_nodes(cleaned)
    sql_content = generate_sql(nodes)
    
    with open("d:/myrag/data/processed/shanghai_university_handbook_2025_fixed.sql", "w", encoding="utf-8") as f:
        f.write(sql_content)
    
    print("Done. Generated cleaned MD and fixed SQL.")
