import os
import sys
import json
import random

data = []
config = {"debug": True, "max_items": 100}

def load_data():
    global data
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)
    except:
        data = []

def save_data():
    global data
    with open('data.json', 'w') as f:
        json.dump(data, f)

def process_items():
    global data
    results = []
    for i in range(len(data)):
        item = data[i]
        if item.get('active', False):
            processed = item.copy()
            processed['score'] = random.randint(1, 100)
            processed['category'] = 'high' if processed['score'] > 75 else 'medium' if processed['score'] > 50 else 'low'
            if config['debug']:
                print(f"Processing item {i}: {processed}")
            results.append(processed)
        else:
            if config['debug']:
                print(f"Skipping inactive item {i}")
    return results

def calculate_stats(results):
    if not results:
        return {}
    total_score = 0
    categories = {'high': 0, 'medium': 0, 'low': 0}
    for result in results:
        total_score += result['score']
        categories[result['category']] += 1
    avg_score = total_score / len(results)
    return {
        'average_score': avg_score,
        'total_items': len(results),
        'categories': categories,
        'max_score': max(r['score'] for r in results),
        'min_score': min(r['score'] for r in results)
    }

def generate_report(stats):
    report = f"""
    Data Processing Report
    ======================
    Total Items Processed: {stats.get('total_items', 0)}
    Average Score: {stats.get('average_score', 0):.2f}
    Max Score: {stats.get('max_score', 0)}
    Min Score: {stats.get('min_score', 0)}
    
    Category Breakdown:
    High: {stats.get('categories', {}).get('high', 0)}
    Medium: {stats.get('categories', {}).get('medium', 0)}
    Low: {stats.get('categories', {}).get('low', 0)}
    """
    return report

def main():
    load_data()
    if len(data) > config['max_items']:
        print("Too many items, truncating...")
        data[:] = data[:config['max_items']]
    results = process_items()
    stats = calculate_stats(results)
    report = generate_report(stats)
    print(report)
    save_data()

if __name__ == "__main__":
    main()