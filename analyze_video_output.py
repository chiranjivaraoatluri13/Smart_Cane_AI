"""Analyze the video processing output to understand why STOP is being triggered."""

import json
from pathlib import Path
from collections import Counter

logs_path = Path("output/video_debug/logs.json")

if not logs_path.exists():
    print("❌ logs.json not found")
    exit(1)

with open(logs_path) as f:
    logs = json.load(f)

print(f"📊 Analyzing {len(logs)} frames\n")

# Count commands
commands = Counter(log.get("command") for log in logs if "command" in log)
print("📈 Command distribution:")
for cmd, count in commands.most_common():
    pct = (count / len(logs)) * 100
    print(f"   {cmd}: {count} frames ({pct:.1f}%)")

print("\n🔍 Analyzing first 10 frames in detail:\n")

for i, log in enumerate(logs[:10]):
    if "facts" not in log:
        continue
    
    facts = log["facts"]
    print(f"Frame {i}:")
    print(f"  Command: {log['command']}")
    print(f"  Phrase: {log.get('phrase', 'N/A')}")
    print(f"  Vision Stop: {facts.get('vision_stop')}")
    print(f"  Walkable by side: {facts.get('walkable_by_side')}")
    print(f"  Hazards by side: {facts.get('hazards_by_side')}")
    print()

print("\n⚠️  ISSUE ANALYSIS:\n")

# Check walkable ratios
walkable_ratios = []
for log in logs:
    if "facts" in log:
        walkable = log["facts"].get("walkable_by_side", {})
        if walkable:
            avg_walkable = sum(walkable.values()) / len(walkable)
            walkable_ratios.append(avg_walkable)

if walkable_ratios:
    avg = sum(walkable_ratios) / len(walkable_ratios)
    print(f"Average walkable ratio: {avg:.2%}")
    print(f"Min walkable ratio: {min(walkable_ratios):.2%}")
    print(f"Max walkable ratio: {max(walkable_ratios):.2%}")
    
    if avg < 0.1:
        print("\n❌ PROBLEM: Walkable ratio is very low!")
        print("   The segmenter is not detecting the footpath as walkable.")
        print("   This causes vision_stop=true on every frame.")
else:
    print("❌ No walkable ratio data found")

print("\n💡 POSSIBLE CAUSES:")
print("   1. Segmentation model not recognizing the surface as 'road' or 'sidewalk'")
print("   2. Camera angle pointing down instead of ahead")
print("   3. Lighting conditions affecting segmentation")
print("   4. Model trained on different surface types")

print("\n✅ NEXT STEPS:")
print("   1. Check the segmentation overlay in output frames")
print("   2. Verify camera is pointing at the path ahead (not down)")
print("   3. Test with different lighting conditions")
print("   4. Check if the surface is being classified as 'building' or 'wall'")
