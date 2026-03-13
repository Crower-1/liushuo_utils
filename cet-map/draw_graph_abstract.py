import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure sized for double-column (6.85" width, 3.5" height) at 300 DPI
fig, ax = plt.subplots(figsize=(6.85, 3.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Define palette
colors = {
    'bg': '#EEECE0',
    'dark': '#38364C',
    'accent1': '#B0CCD0',
    'accent2': '#72909F',
    'accent3': '#7B7A8E',
}

# Background
ax.add_patch(patches.Rectangle((0, 0), 100, 100, color=colors['bg']))

# Left: Challenges
ax.text(5, 95, "Challenges", fontsize=14, fontweight='bold', color=colors['dark'])
challenge_boxes = [
    ("I. 单一模型难扩展", 80),
    ("II. 深度学习落地难", 55),
    ("III. 无公开社区", 30)
]
for text, y in challenge_boxes:
    ax.add_patch(patches.Rectangle((5, y-5), 28, 12, edgecolor=colors['dark'], facecolor='white'))
    ax.text(6, y+3, text, fontsize=10, color=colors['dark'])

# Middle: CET-MAP Solution
ax.text(40, 95, "CET-MAP Solution", fontsize=14, fontweight='bold', color=colors['dark'])
solution_boxes = [
    ("Foundation Model", (40, 75)),
    ("GUI & Pipeline", (40, 50)),
    ("Community", (40, 25))
]
for label, (x, y) in solution_boxes:
    ax.add_patch(patches.Rectangle((x, y-5), 28, 12, edgecolor=colors['accent2'], facecolor=colors['accent1']))
    ax.text(x+1, y+3, label, fontsize=10, color=colors['dark'])

# Arrows linking challenges to solution
ax.annotate('', xy=(33, 86), xytext=(40, 82),
            arrowprops=dict(arrowstyle="->", color=colors['dark']))
ax.annotate('', xy=(33, 61), xytext=(40, 56),
            arrowprops=dict(arrowstyle="->", color=colors['dark']))
ax.annotate('', xy=(33, 36), xytext=(40, 31),
            arrowprops=dict(arrowstyle="->", color=colors['dark']))

# Right: Community details
ax.text(75, 95, "Public Datasets", fontsize=14, fontweight='bold', color=colors['dark'])
public_boxes = [
    ("Model Dataset\n- 低存储成本\n- 可扩展", 70),
    ("Label Dataset\n- 轻量标签\n- 统一索引", 40)
]
for text, y in public_boxes:
    ax.add_patch(patches.Rectangle((60, y-6), 28, 16, edgecolor=colors['accent3'], facecolor='white'))
    ax.text(61, y+4, text, fontsize=9, color=colors['dark'])

# Loop arrow at bottom
ax.annotate('', xy=(90, 10), xytext=(10, 10),
            arrowprops=dict(arrowstyle="->", linestyle='--', color=colors['dark']))
ax.text(30, 5, "更多数据 → 优化 Foundation Model", fontsize=9, color=colors['dark'])

# Save as high-resolution PNG
plt.savefig('/home/liushuo/Documents/paper/SynapseSeg/CET_MAP_graphical_abstract.png', bbox_inches='tight', dpi=300)
plt.close()


