import re

# Sample text containing parameter values and lines with layer information
text = """
Graph0   Input: 63.8155   Task: 680.745   send: 0.0030044   Out: 1.06234   Process: 745.626
Adding Graph0 target 1 PE: B Host PE: B Layers: 0-7
Graph1   Input: 50.1234   Task: 550.678   send: 0.0023456   Out: 0.98765   Process: 600.123
Adding Graph1 target 2 PE: A Host PE: A Layers: 8-15
Graph2   Input: 70.9876   Task: 750.432   send: 0.0045678   Out: 1.23456   Process: 800.987
Adding Graph2 target 3 PE: C Host PE: C Layers: 16-23
"""

# Define a regular expression pattern to match the lines with layer information
pattern = r'Adding Graph(\d+)\s+target \d+ PE: [A-Z]\s+Host PE: [A-Z]\s+Layers: (\d+)-(\d+)'

# Find all matches in the text
matches = re.findall(pattern, text)

# Iterate through the matches and extract the start and end layers
for match in matches:
    graph_number, start_layer, end_layer = match
    print(f"Graph{graph_number}:")
    print(f"  Start Layer: {start_layer}")
    print(f"  End Layer: {end_layer}")
    print()
