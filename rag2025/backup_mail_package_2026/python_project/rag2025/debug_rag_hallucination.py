"""
Debug Script for RAG Hallucination Issue
Diagnoses why program count queries return wrong results.
"""
import json
from pathlib import Path
from typing import Dict, List, Any

# ===== CONFIGURATION =====
CHUNKED_DATA_PATH = "D:/chunking/rag2025_2/rag2025/data/chunked/chunked_10.jsonl"


# ===== STEP 1: LOAD ALL PROGRAMS FROM CHUNKED DATA =====
def load_programs(file_path: str) -> List[Dict[str, Any]]:
    """Load all program chunks from JSONL file."""
    programs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if data.get('faq_type') == 'thong_tin_nganh':
                        programs.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skip line {line_num}: Invalid JSON - {e}")
                    continue
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {file_path}")
        return []

    print(f"✅ Loaded {len(programs)} program chunks from {len(line_num)} total lines")
    return programs


# ===== STEP 2: SHOW CORRECT ANSWER =====
def show_correct_answer(programs: List[Dict[str, Any]]) -> None:
    """Display the correct answer based on chunked data."""
    print("\n" + "=" * 70)
    print("CORRECT ANSWER (from chunked data)")
    print("=" * 70)

    # Group by category
    categories = {
        'Công nghệ thông tin': [],
        'Kỹ thuật & Công nghệ': [],
        'Khoa học': [],
        'Khoa học xã hội & Nhân văn': [],
        'Khác': []
    }

    for prog in programs:
        meta = prog.get('metadata', {})
        name = meta.get('name', '')

        # Categorize
        if any(x in name.lower() for x in ['công nghệ thông tin', 'phần mềm', 'dữ liệu']):
            categories['Công nghệ thông tin'].append(prog)
        elif any(x in name.lower() for x in ['điện tử', 'viễn thông', 'sinh học', 'hóa học', 'trắc địa', 'bản đồ', 'xây dựng']):
            categories['Kỹ thuật & Công nghệ'].append(prog)
        elif any(x in name.lower() for x in ['vật lý', 'hóa học', 'sinh học', 'toán học', 'môi trường']):
            categories['Khoa học'].append(prog)
        elif any(x in name.lower() for x in ['lịch sử', 'văn học', 'triết học', 'báo chí', 'truyền thông', 'hán nôm', 'đông phương', 'khảo cổ', 'ngữ văn']):
            categories['Khoa học xã hội & Nhân văn'].append(prog)
        else:
            categories['Khác'].append(prog)

    # Display
    total = 0
    for category, progs in categories.items():
        if progs:
            print(f"\n📌 {category} ({len(progs)} ngành):")
            for prog in progs:
                meta = prog.get('metadata', {})
                print(f"   • {meta.get('name', '')} (Mã: {meta.get('program_code', '')})")
                print(f"     - Chỉ tiêu: {meta.get('quota', 'N/A')}")
                print(f"     - Tổ hợp: {', '.join(meta.get('combinations', []))}")
                total += 1

    print(f"\n✅ Tổng cộng: {total} ngành")

    # Check for blacklisted majors
    blacklist = ['y khoa', 'dược', 'răng', 'hàm', 'mặt', 'kinh tế', 'quản trị kinh doanh', 'luật', 'giáo dục']
    found_blacklist = False

    print("\n" + "-" * 70)
    print("BLACKLIST CHECK (these majors DON'T exist at ĐHKH Hue)")
    print("-" * 70)

    for prog in programs:
        meta = prog.get('metadata', {})
        name = meta.get('name', '').lower()

        if any(bad in name for bad in blacklist):
            print(f"⚠️  WARNING: Found potential blacklist term: {meta.get('name', '')}")
            print(f"     This might indicate incorrect data!")
            found_blacklist = True

    if not found_blacklist:
        print("✅ No blacklist terms found in program data")


# ===== STEP 3: TEST HYDE QUERY EXPANSION =====
def test_hyde_expansion(query: str) -> List[str]:
    """Simulate HYDE query expansion."""
    print("\n" + "=" * 70)
    print("HYDE QUERY EXPANSION TEST")
    print("=" * 70)
    print(f"Original query: '{query}'")
    print("\nExpected HYDE variants:")

    expected_variants = [
        "Danh sách các ngành đào tạo tại Đại học Khoa học Huế",
        "Thông tin tuyển sinh các ngành ĐHKH Huế 2025",
        "Chỉ tiêu tuyển sinh các ngành Đại học Khoa học Huế",
        "Tổng số ngành học tại trường Đại học Khoa học Huế",
        "Các ngành đào tạo ĐHKH Huế năm 2025",
    ]

    for i, variant in enumerate(expected_variants, 1):
        print(f"  Variant {i}: {variant}")

    return expected_variants


# ===== STEP 4: CHECK CHUNK CONTENT =====
def analyze_chunk_content(programs: List[Dict[str, Any]]) -> None:
    """Analyze chunk content structure."""
    print("\n" + "=" * 70)
    print("CHUNK CONTENT ANALYSIS")
    print("=" * 70)

    for i, prog in enumerate(programs[:3], 1):  # Show first 3
        print(f"\nChunk {i} Sample:")
        print(f"  ID: {prog.get('id', '')}")
        print(f"  FAQ type: {prog.get('faq_type', '')}")

        meta = prog.get('metadata', {})
        print(f"  Program name: {meta.get('name', '')}")
        print(f"  Program code: {meta.get('program_code', '')}")
        print(f"  Year: {meta.get('year', '')}")
        print(f"  Quota: {meta.get('quota', '')}")

        print(f"\n  Text length: {len(prog.get('text', ''))} chars")
        print(f"  Summary length: {len(prog.get('summary', ''))} chars")
        print(f"  Text_plain length: {len(prog.get('text_plain', ''))} chars")

        # Check overlap
        text = prog.get('text', '')
        summary = prog.get('summary', '')
        text_plain = prog.get('text_plain', '')

        if summary:
            overlap = sum(1 for word in summary.split() if word in text.lower()) / len(summary.split()) if len(summary.split()) > 0 else 0
            print(f"  Summary vs Text overlap: {overlap*100:.1f}%")


# ===== STEP 5: VERIFICATION CHECKLIST =====
def print_verification_checklist() -> None:
    """Print verification checklist for debugging."""
    print("\n" + "=" * 70)
    print("VERIFICATION CHECKLIST")
    print("=" * 70)
    print("""
Before trusting LLM response, check:

[ ] Are all listed majors in the retrieved chunks?
[ ] Does the count match the number of program chunks?
[ ] Are any blacklisted majors mentioned (Y, Dược, Kinh tế, Luật)?
[ ] Is the response based ONLY on context or using general knowledge?
[ ] Does the response say "khoảng" instead of exact number?

If ANY answer is NO → LLM is hallucinating!

Recommended Actions:
1. Add metadata filter: {"faq_type": "thong_tin_nganh", "year": 2025}
2. Use higher top_k for program list queries (e.g., 30 instead of 5)
3. Add explicit blacklist to generation prompt
4. Verify retrieval returns program chunks
    """)


# ===== MAIN DIAGNOSTIC FLOW =====
def main():
    """Main diagnostic flow."""
    print("\n" + "=" * 70)
    print("RAG SYSTEM DEBUGGER - HALLUCINATION ISSUE")
    print("=" * 70)
    print("Query: 'ĐHKH Huế có bao nhiêu ngành?'")

    # Step 1: Load programs
    programs = load_programs(CHUNKED_DATA_PATH)

    if not programs:
        print("\n❌ No programs loaded. Cannot continue.")
        return

    # Step 2: Show correct answer
    show_correct_answer(programs)

    # Step 3: Test HYDE expansion
    test_hyde_expansion("ĐHKH Huế có bao nhiêu ngành?")

    # Step 4: Analyze chunk content
    analyze_chunk_content(programs)

    # Step 5: Print verification checklist
    print_verification_checklist()

    # Step 6: Summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print(f"""
📊 DATA STATS:
  - Total program chunks: {len(programs)}
  - Unique program codes: {len(set(p.get('metadata', {}).get('program_code', '') for p in programs))}
  - Year: {programs[0].get('metadata', {}).get('year', 'N/A') if programs else 'N/A'}

🔍 DIAGNOSIS:
  1. Check if retrieval is filtering by faq_type='thong_tin_nganh'
  2. Verify HYDE generates semantic variants matching program info
  3. Ensure generation prompt has anti-hallucination rules
  4. Test if response mentions blacklisted majors

✅ FIXES IMPLEMENTED:
  [x] Generation prompt updated with blacklist rules
  [x] Generation prompt updated with verification checklist
  [x] Retrieval service supports metadata filtering
  [x] main.py detects program list queries
  [x] main.py applies metadata filter for program queries
  [ ] Higher top_k for program list queries (set to top_k * 3)

🎯 NEXT STEPS:
1. Run: python debug_rag_hallucination.py
2. Test query: "ĐHKH Huế có bao nhiêu ngành?"
3. Check logs for:
   - "program_list_query: True"
   - "metadata_filter: {'faq_type': 'thong_tin_nganh', 'year': 2025}"
   - Number of retrieved chunks (should be >5 for program list)
4. Verify response doesn't include blacklisted majors
    """)

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
