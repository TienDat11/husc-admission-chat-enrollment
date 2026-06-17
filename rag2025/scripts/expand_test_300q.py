"""
Expand test to ~300 questions targeting RAG system weaknesses.

Adds ~180 questions across 7 weakness categories:
1. Retrieval gaps (missing chunks)
2. Multi-entity fusion (3+ chunks needed)
3. Enumeration completeness (must list ALL items)
4. Negation/exclusion (what is NOT available)
5. Context conflict (contradictory/ambiguous info)
6. Conversational/imprecise (real student phrasing)
7. Cross-year comparison (temporal reasoning)

Also adds weakness_type field to all existing questions.

Usage:
    cd rag2025
    python scripts/expand_test_300q.py
"""
from __future__ import annotations

import json
from pathlib import Path

TEST_PATH = Path(__file__).parent.parent / "data" / "eval" / "husc_86q_test.json"

# Weakness-type mapping for existing questions
EXISTING_WEAKNESS_MAP = {
    "simple": "single_fact",
    "multihop": "multi_entity",
    "temporal": "temporal",
}

NEW_QUESTIONS = [
    # ==========================================
    # CATEGORY 1: RETRIEVAL GAPS (40 questions)
    # ==========================================
    {"id":"q124","category":"simple","question":"Mã ngành Báo chí HUSC là gì?","ground_truth_answer":"7320101","gt_chunks":["tuyensinh_baochi_2026"],"must_include":["7320101"],"weakness_type":"retrieval_gap"},
    {"id":"q125","category":"simple","question":"Mã ngành Luật của HUSC là gì?","ground_truth_answer":"7380101","gt_chunks":["tuyensinh_luat_2026"],"must_include":["7380101"],"weakness_type":"retrieval_gap"},
    {"id":"q126","category":"simple","question":"Mã ngành Triết học HUSC?","ground_truth_answer":"7229001","gt_chunks":["tuyensinh_trieuthoc_2026"],"must_include":["7229001"],"weakness_type":"retrieval_gap"},
    {"id":"q127","category":"simple","question":"Mã ngành Xã hội học HUSC?","ground_truth_answer":"7310101","gt_chunks":["tuyensinh_xahoihoc_2026"],"must_include":["7310101"],"weakness_type":"retrieval_gap"},
    {"id":"q128","category":"simple","question":"Mã ngành Đông phương học HUSC?","ground_truth_answer":"7310201","gt_chunks":["tuyensinh_dongphuong_2026"],"must_include":["7310201"],"weakness_type":"retrieval_gap"},
    {"id":"q129","category":"simple","question":"Mã ngành Kiến trúc HUSC?","ground_truth_answer":"7580101","gt_chunks":["tuyensinh_kientruc_2026"],"must_include":["7580101"],"weakness_type":"retrieval_gap"},
    {"id":"q130","category":"simple","question":"Mã ngành Quản lý công HUSC?","ground_truth_answer":"7340101","gt_chunks":["tuyensinh_quanlycong_2026"],"must_include":["7340101"],"weakness_type":"retrieval_gap"},
    {"id":"q131","category":"simple","question":"Mã ngành Chính trị học HUSC?","ground_truth_answer":"7310101","gt_chunks":["tuyensinh_chinhtri_2026"],"must_include":["Chính trị"],"weakness_type":"retrieval_gap"},
    {"id":"q132","category":"simple","question":"Chỉ tiêu ngành Báo chí HUSC 2026?","ground_truth_answer":"60 chỉ tiêu","gt_chunks":["tuyensinh_baochi_2026"],"must_include":["60"],"weakness_type":"retrieval_gap"},
    {"id":"q133","category":"simple","question":"Chỉ tiêu ngành Luật HUSC 2026?","ground_truth_answer":"60 chỉ tiêu","gt_chunks":["tuyensinh_luat_2026"],"must_include":["60"],"weakness_type":"retrieval_gap"},
    {"id":"q134","category":"simple","question":"Chỉ tiêu ngành Kiến trúc HUSC 2026?","ground_truth_answer":"60 chỉ tiêu","gt_chunks":["tuyensinh_kientruc_2026"],"must_include":["60"],"weakness_type":"retrieval_gap"},
    {"id":"q135","category":"simple","question":"Học phí khối ngành IV HUSC 2026?","ground_truth_answer":"540.000 VNĐ/tín chỉ","gt_chunks":["tuyensinh_overview_2026"],"must_include":["540000"],"weakness_type":"retrieval_gap"},
    {"id":"q136","category":"simple","question":"Học phí khối ngành VII HUSC 2026?","ground_truth_answer":"525.000 VNĐ/tín chỉ","gt_chunks":["tuyensinh_overview_2026"],"must_include":["525000"],"weakness_type":"retrieval_gap"},
    {"id":"q137","category":"simple","question":"Điểm chuẩn ngành Hóa học HUSC 2026?","ground_truth_answer":"18.0","gt_chunks":["tuyensinh_hoahoc_2026"],"must_include":["18"],"weakness_type":"retrieval_gap"},
    {"id":"q138","category":"simple","question":"Điểm chuẩn ngành Sinh học HUSC 2026?","ground_truth_answer":"17.5","gt_chunks":["tuyensinh_sinhhoc_2026"],"must_include":["17"],"weakness_type":"retrieval_gap"},
    {"id":"q139","category":"simple","question":"Điểm chuẩn ngành Địa lý HUSC 2026?","ground_truth_answer":"18.0","gt_chunks":["tuyensinh_dialy_2026"],"must_include":["18"],"weakness_type":"retrieval_gap"},
    {"id":"q140","category":"simple","question":"Điểm chuẩn ngành CNTT HUSC 2026?","ground_truth_answer":"20.5","gt_chunks":["tuyensinh_cntt_2026"],"must_include":["20"],"weakness_type":"retrieval_gap"},
    {"id":"q141","category":"simple","question":"Số điện thoại tư vấn tuyển sinh HUSC?","ground_truth_answer":"094 4455136","gt_chunks":["husc_info"],"must_include":["094"],"weakness_type":"retrieval_gap"},
    {"id":"q142","category":"simple","question":"Mã trường Đại học Khoa học Huế trên hệ thống Bộ GDĐT?","ground_truth_answer":"DHT","gt_chunks":["husc_info"],"must_include":["DHT"],"weakness_type":"retrieval_gap"},
    {"id":"q143","category":"simple","question":"Địa chỉ nộp hồ sơ xét tuyển HUSC?","ground_truth_answer":"01 Điện Biên Phủ, phường Vĩnh Ninh, TP. Huế","gt_chunks":["husc_info"],"must_include":["Điện Biên Phủ"],"weakness_type":"retrieval_gap"},
    {"id":"q144","category":"simple","question":"Tổ hợp xét tuyển ngành Báo chí HUSC?","ground_truth_answer":"C00, C01, C19, D01","gt_chunks":["tuyensinh_baochi_2026"],"must_include":["C00"],"weakness_type":"retrieval_gap"},
    {"id":"q145","category":"simple","question":"Tổ hợp xét tuyển ngành Luật HUSC?","ground_truth_answer":"A00, C00, C19, D01","gt_chunks":["tuyensinh_luat_2026"],"must_include":["A00","C00"],"weakness_type":"retrieval_gap"},
    {"id":"q146","category":"simple","question":"Tổ hợp xét tuyển ngành Triết học HUSC?","ground_truth_answer":"C00, C19, D01","gt_chunks":["tuyensinh_trieuthoc_2026"],"must_include":["C00"],"weakness_type":"retrieval_gap"},
    {"id":"q147","category":"simple","question":"Thời gian đào tạo ngành Kiến trúc HUSC?","ground_truth_answer":"5 năm","gt_chunks":["tuyensinh_kientruc_2026"],"must_include":["5","năm"],"weakness_type":"retrieval_gap"},
    {"id":"q148","category":"simple","question":"Điểm vẽ mỹ thuật tối thiểu cho ngành Kiến trúc?","ground_truth_answer":"5.0 điểm","gt_chunks":["tuyensinh_kientruc_2026"],"must_include":["5.0"],"weakness_type":"retrieval_gap"},
    {"id":"q149","category":"simple","question":"Chỉ tiêu tuyển sinh thẳng tối đa bao nhiêu % cho mỗi ngành?","ground_truth_answer":"10%","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["10%"],"weakness_type":"retrieval_gap"},
    {"id":"q150","category":"simple","question":"Hạn nộp lệ phí xét tuyển đợt 1 năm 2026?","ground_truth_answer":"Từ 15/7 đến 17h00 ngày 21/7/2026","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["lệ phí"],"weakness_type":"retrieval_gap"},
    {"id":"q151","category":"simple","question":"Thí sinh được đăng ký tối đa bao nhiêu nguyện vọng?","ground_truth_answer":"15 nguyện vọng","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["15"],"weakness_type":"retrieval_gap"},
    {"id":"q152","category":"simple","question":"Ngành Triết học HUSC có miễn học phí không?","ground_truth_answer":"Có, Triết học được miễn học phí (0đ/tín chỉ)","gt_chunks":["tuyensinh_trieuthoc_2026"],"must_include":["miễn học phí"],"weakness_type":"retrieval_gap"},
    {"id":"q153","category":"simple","question":"Hạn sử dụng chứng chỉ tiếng Anh để quy đổi điểm?","ground_truth_answer":"Còn hạn đến 12/08/2026","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["12/08"],"weakness_type":"retrieval_gap"},
    {"id":"q154","category":"simple","question":"Thí sinh khu vực KV1 được cộng bao nhiêu điểm ưu tiên?","ground_truth_answer":"0.75 điểm","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["0.75"],"weakness_type":"retrieval_gap"},
    {"id":"q155","category":"simple","question":"Thí sinh KV2-NT được cộng bao nhiêu điểm?","ground_truth_answer":"0.5 điểm","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["0.5"],"weakness_type":"retrieval_gap"},
    {"id":"q156","category":"simple","question":"Giải Nhất HSG cấp tỉnh được cộng bao nhiêu điểm thưởng?","ground_truth_answer":"3.0 điểm","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["3.0"],"weakness_type":"retrieval_gap"},
    {"id":"q157","category":"simple","question":"Giải Khuyến khích HSG cấp tỉnh được cộng bao nhiêu?","ground_truth_answer":"1.5 điểm","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["1.5"],"weakness_type":"retrieval_gap"},
    {"id":"q158","category":"simple","question":"Thủ khoa HUSC được miễn học phí bao nhiêu học kỳ?","ground_truth_answer":"Miễn 100% học phí học kỳ 1","gt_chunks":["hocbong_2026"],"must_include":["100%"],"weakness_type":"retrieval_gap"},
    {"id":"q159","category":"simple","question":"Link đăng ký ký túc xá HUSC?","ground_truth_answer":"http://sv-dhh.kytucxa.vn/","gt_chunks":["husc_info"],"must_include":["kytucxa"],"weakness_type":"retrieval_gap"},
    {"id":"q160","category":"simple","question":"Ngành Sư phạm Tin học mã ngành bao nhiêu?","ground_truth_answer":"7140212","gt_chunks":["tuyensinh_sptinhoc_2026"],"must_include":["7140212"],"weakness_type":"retrieval_gap"},
    {"id":"q161","category":"simple","question":"Ngành Sư phạm Toán học mã ngành bao nhiêu?","ground_truth_answer":"7140202","gt_chunks":["tuyensinh_sptoan_2026"],"must_include":["7140202"],"weakness_type":"retrieval_gap"},
    {"id":"q162","category":"simple","question":"IELTS 7.0 quy đổi thành bao nhiêu điểm thi THPT môn Tiếng Anh?","ground_truth_answer":"10.0 điểm","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["10.0"],"weakness_type":"retrieval_gap"},
    {"id":"q163","category":"simple","question":"VSTEP B2 quy đổi thành bao nhiêu điểm tiếng Anh?","ground_truth_answer":"8.0-9.0 điểm","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["VSTEP","8"],"weakness_type":"retrieval_gap"},

    # ==========================================
    # CATEGORY 2: MULTI-ENTITY FUSION (35 questions)
    # ==========================================
    {"id":"q164","category":"multihop","question":"So sánh học phí, chỉ tiêu và điểm chuẩn giữa ngành CNTT và Kỹ thuật phần mềm.","ground_truth_answer":"Cả hai thuộc khối V, học phí ~600000đ/tín chỉ. CNTT chỉ tiêu 510, KTPM chỉ tiêu thấp hơn.","gt_chunks":["tuyensinh_cntt_2026","tuyensinh_ktpm_2026"],"must_include":["600000","CNTT","Kỹ thuật phần mềm"],"weakness_type":"multi_entity"},
    {"id":"q165","category":"multihop","question":"Ngành nào có học phí thấp nhất và cao nhất tại HUSC? Nêu mức học phí cụ thể.","ground_truth_answer":"Triết học miễn học phí (0đ, thấp nhất). Khối V (CNTT, KTPM, Kiến trúc) cao nhất ~600000đ/tín chỉ.","gt_chunks":["tuyensinh_overview_2026"],"must_include":["Triết","600000"],"weakness_type":"multi_entity"},
    {"id":"q166","category":"multihop","question":"Liệt kê tất cả ngành có chỉ tiêu trên 80 sinh viên tại HUSC 2026.","ground_truth_answer":"CNTT 510, KTPM, KHMT, Tin học 120, Toán học 100, Hóa học 90, Vật lý 80","gt_chunks":["tuyensinh_overview_2026"],"must_include":["510","120","100","90"],"weakness_type":"multi_entity"},
    {"id":"q167","category":"multihop","question":"Những ngành nào xét tuyển tổ hợp A00 tại HUSC?","ground_truth_answer":"Toán học, Vật lý, Hóa học, CNTT, KTPM, KHMT, Tin học, Luật và một số ngành khác","gt_chunks":["tuyensinh_overview_2026"],"must_include":["A00","CNTT","Toán"],"weakness_type":"multi_entity"},
    {"id":"q168","category":"multihop","question":"Những ngành nào có học phí 600000 VNĐ/tín chỉ?","ground_truth_answer":"Các ngành khối V: CNTT, KTPM, KHMT, Kiến trúc, Tin học","gt_chunks":["tuyensinh_overview_2026"],"must_include":["600000","CNTT","Kiến trúc"],"weakness_type":"multi_entity"},
    {"id":"q169","category":"multihop","question":"Những ngành nào có học phí 525000 VNĐ/tín chỉ?","ground_truth_answer":"Các ngành khối VII: Văn học, Lịch sử, Báo chí, Triết học, Xã hội học, Đông phương học","gt_chunks":["tuyensinh_overview_2026"],"must_include":["525000","Văn học","Lịch sử"],"weakness_type":"multi_entity"},
    {"id":"q170","category":"multihop","question":"So sánh chỉ tiêu tuyển sinh giữa khoa Toán và khoa CNTT.","ground_truth_answer":"Khoa Toán: Toán học 100, Tin học 120. Khoa CNTT: CNTT 510, KTPM, KHMT. Khoa CNTT có chỉ tiêu cao hơn nhiều.","gt_chunks":["tuyensinh_toan_2026","tuyensinh_cntt_2026"],"must_include":["510","100","120"],"weakness_type":"multi_entity"},
    {"id":"q171","category":"multihop","question":"Ngành nào vừa có tổ hợp C00 vừa có chỉ tiêu trên 50?","ground_truth_answer":"Văn học, Lịch sử, Báo chí, Luật, Triết học, Xã hội học, Đông phương học đều có C00 và chỉ tiêu từ 50 trở lên","gt_chunks":["tuyensinh_overview_2026"],"must_include":["C00","50"],"weakness_type":"multi_entity"},
    {"id":"q172","category":"multihop","question":"So sánh tổ hợp xét tuyển giữa ngành Văn học và Báo chí.","ground_truth_answer":"Văn học: C00, D01. Báo chí: C00, C01, C19, D01. Báo chí có nhiều tổ hợp hơn.","gt_chunks":["tuyensinh_vanhoc_2026","tuyensinh_baochi_2026"],"must_include":["Văn học","Báo chí","C00"],"weakness_type":"multi_entity"},
    {"id":"q173","category":"multihop","question":"Tổng chỉ tiêu tuyển sinh của tất cả ngành khối khoa học tự nhiên là bao nhiêu?","ground_truth_answer":"Toán 100 + Tin 120 + Lý 80 + Hóa 90 + Sinh 70 = khoảng 460 chỉ tiêu","gt_chunks":["tuyensinh_overview_2026"],"must_include":["460","chỉ tiêu"],"weakness_type":"multi_entity"},
    {"id":"q174","category":"multihop","question":"So sánh điểm chuẩn giữa các ngành khoa học xã hội (Văn, Sử, Địa) năm 2026.","ground_truth_answer":"Văn học, Lịch sử, Địa lý thường có điểm chuẩn tương đương khoảng 18-19 điểm","gt_chunks":["tuyensinh_vanhoc_2026","tuyensinh_ls_2026","tuyensinh_dialy_2026"],"must_include":["Văn","Lịch sử","Địa lý"],"weakness_type":"multi_entity"},
    {"id":"q175","category":"multihop","question":"Ngành nào có cả tổ hợp B00 và D07?","ground_truth_answer":"Hóa học và Sinh học đều có tổ hợp B00 và D07","gt_chunks":["tuyensinh_hoahoc_2026","tuyensinh_sinhhoc_2026"],"must_include":["B00","D07","Hóa","Sinh"],"weakness_type":"multi_entity"},
    {"id":"q176","category":"multihop","question":"Liệt kê các ngành có điểm chuẩn dưới 19 tại HUSC 2026.","ground_truth_answer":"Toán học 18.5, Hóa học 18, Sinh học 17.5, Vật lý 16.5, Lịch sử, Địa lý, Xã hội học, Đông phương học","gt_chunks":["tuyensinh_overview_2026"],"must_include":["18","19"],"weakness_type":"multi_entity"},
    {"id":"q177","category":"multihop","question":"Những ngành nào thuộc khối ngành IV (Khoa học tự nhiên) tại HUSC?","ground_truth_answer":"Vật lý học, Hóa học, Sinh học, Khoa học môi trường. Học phí ~540000đ/tín chỉ","gt_chunks":["tuyensinh_overview_2026"],"must_include":["Vật lý","Hóa học","Sinh học","540000"],"weakness_type":"multi_entity"},
    {"id":"q178","category":"multihop","question":"Những ngành nào thuộc khối ngành VII (Khoa học xã hội) tại HUSC?","ground_truth_answer":"Văn học, Lịch sử, Triết học, Xã hội học, Đông phương học, Báo chí. Học phí ~525000đ/tín chỉ","gt_chunks":["tuyensinh_overview_2026"],"must_include":["Văn học","Lịch sử","525000"],"weakness_type":"multi_entity"},
    {"id":"q179","category":"multihop","question":"So sánh học phí giữa 3 khối ngành IV, V, VII tại HUSC 2026.","ground_truth_answer":"Khối IV ~540000đ/tín chỉ, khối V ~600000đ/tín chỉ, khối VII ~525000đ/tín chỉ. Khối V cao nhất.","gt_chunks":["tuyensinh_overview_2026"],"must_include":["540000","600000","525000"],"weakness_type":"multi_entity"},
    {"id":"q180","category":"multihop","question":"Tổng học phí 1 năm cho ngành Văn học và ngành CNTT khác nhau bao nhiêu?","ground_truth_answer":"Văn học: ~525000x30=15750000đ/năm. CNTT: ~600000x30=18000000đ/năm. Chênh 2250000đ","gt_chunks":["tuyensinh_vanhoc_2026","tuyensinh_cntt_2026"],"must_include":["525000","600000"],"weakness_type":"multi_entity"},
    {"id":"q181","category":"multihop","question":"Ngành nào có nhiều tổ hợp xét tuyển nhất tại HUSC?","ground_truth_answer":"CNTT thường có nhiều tổ hợp: A00, A01, C01, D07","gt_chunks":["tuyensinh_cntt_2026"],"must_include":["CNTT","tổ hợp"],"weakness_type":"multi_entity"},
    {"id":"q182","category":"multihop","question":"Có bao nhiêu ngành sử dụng tổ hợp C00 tại HUSC?","ground_truth_answer":"Ít nhất 7 ngành: Văn học, Lịch sử, Báo chí, Luật, Triết học, Xã hội học, Đông phương học","gt_chunks":["tuyensinh_overview_2026"],"must_include":["C00","7"],"weakness_type":"multi_entity"},
    {"id":"q183","category":"multihop","question":"So sánh điều kiện xét tuyển học bạ và thi THPT tại HUSC.","ground_truth_answer":"Học bạ (mã 200): điểm TB 4 học kỳ lớp 11-12. Thi THPT (mã 100): tổng điểm 3 môn. Cả hai đều được cộng điểm ưu tiên và thưởng.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["học bạ","THPT","ưu tiên"],"weakness_type":"multi_entity"},
    {"id":"q184","category":"multihop","question":"HUSC có những phương thức xét tuyển nào? Liệt kê tất cả.","ground_truth_answer":"1) Thi THPT (mã 100), 2) Xét học bạ THPT (mã 200), 3) Xét tuyển thẳng (mã 301), 4) Đánh giá năng lực, 5) Thi năng khiếu (Kiến trúc)","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["100","200","301"],"weakness_type":"multi_entity"},
    {"id":"q185","category":"multihop","question":"So sánh ngành Sư phạm Toán và Toán học về tổ hợp và học phí.","ground_truth_answer":"Sư phạm Toán và Toán học đều thuộc HUSC, nhưng Sư phạm Toán có thể được miễn học phí theo chính sách sư phạm. Tổ hợp có thể khác nhau.","gt_chunks":["tuyensinh_sptoan_2026","tuyensinh_toan_2026"],"must_include":["Sư phạm","Toán"],"weakness_type":"multi_entity"},
    {"id":"q186","category":"multihop","question":"Ngành nào có điểm chuẩn cao nhất và thấp nhất tại HUSC 2026?","ground_truth_answer":"CNTT thường cao nhất (~20.5), Vật lý thường thấp nhất (~16.5)","gt_chunks":["tuyensinh_overview_2026"],"must_include":["CNTT","Vật lý"],"weakness_type":"multi_entity"},
    {"id":"q187","category":"multihop","question":"Liệt kê tất cả ngành có xét tuyển bằng thi năng khiếu tại HUSC.","ground_truth_answer":"Kiến trúc (V00, V01, V02, V12) - thi Vẽ mỹ thuật","gt_chunks":["tuyensinh_kientruc_2026"],"must_include":["Kiến trúc","Vẽ mỹ thuật"],"weakness_type":"multi_entity"},
    {"id":"q188","category":"multihop","question":"Những ngành nào có chỉ tiêu dưới 60 tại HUSC 2026?","ground_truth_answer":"Địa lý 50, Lịch sử 50, và một số ngành khác","gt_chunks":["tuyensinh_overview_2026"],"must_include":["50"],"weakness_type":"multi_entity"},
    {"id":"q189","category":"multihop","question":"Tổng học phí 4 năm cho ngành Hóa học khoảng bao nhiêu?","ground_truth_answer":"Hóa học khối IV: ~540000đ/tín chỉ x ~120 tín chỉ = ~64800000đ/4 năm","gt_chunks":["tuyensinh_hoahoc_2026"],"must_include":["540000","tín chỉ"],"weakness_type":"multi_entity"},
    {"id":"q190","category":"multihop","question":"So sánh điểm chuẩn ngành CNTT và KTPM năm 2026.","ground_truth_answer":"CNTT điểm chuẩn cao hơn KTPM do chỉ tiêu CNTT rất lớn (510) nhưng nhu cầu cũng cao","gt_chunks":["tuyensinh_cntt_2026","tuyensinh_ktpm_2026"],"must_include":["CNTT","Kỹ thuật phần mềm"],"weakness_type":"multi_entity"},
    {"id":"q191","category":"multihop","question":"So sánh điều kiện đầu vào giữa ngành Kiến trúc và các ngành khác.","ground_truth_answer":"Kiến trúc yêu cầu thi Vẽ mỹ thuật (đạt >=5.0), các ngành khác chỉ xét điểm thi THPT hoặc học bạ","gt_chunks":["tuyensinh_kientruc_2026","dieukien_tuyensinh_2026"],"must_include":["Vẽ mỹ thuật","5.0"],"weakness_type":"multi_entity"},
    {"id":"q192","category":"multihop","question":"Có bao nhiêu khối ngành tại HUSC và học phí mỗi khối là bao nhiêu?","ground_truth_answer":"3 khối chính: IV (~540000đ/tín chỉ), V (~600000đ/tín chỉ), VII (~525000đ/tín chỉ). Triết học miễn học phí.","gt_chunks":["tuyensinh_overview_2026"],"must_include":["540000","600000","525000"],"weakness_type":"multi_entity"},
    {"id":"q193","category":"multihop","question":"Ngành CNTT có mấy chuyên ngành? Liệt kê tất cả.","ground_truth_answer":"3 chuyên ngành: Khoa học máy tính (AI), Mạng máy tính, Công nghệ phần mềm","gt_chunks":["tuyensinh_cntt_2026"],"must_include":["Khoa học máy tính","Mạng máy tính","Công nghệ phần mềm"],"weakness_type":"multi_entity"},
    {"id":"q194","category":"multihop","question":"So sánh việc xét tuyển bằng điểm thi THPT và học bạ: cái nào có lợi hơn?","ground_truth_answer":"Tùy điểm mạnh của thí sinh. Thi THPT dùng tổng 3 môn, học bạ dùng điểm TB 4 học kỳ. Có thể đăng ký cả hai phương thức.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["THPT","học bạ"],"weakness_type":"multi_entity"},
    {"id":"q195","category":"multihop","question":"HUSC có những chuyên ngành nào trong ngành CNTT? So sánh các chuyên ngành.","ground_truth_answer":"3 chuyên ngành: KHMT(AI) - trọng tâm AI/ML, Mạng máy tính - trọng tâm mạng/security, CNPM - trọng tâm phát triển phần mềm","gt_chunks":["tuyensinh_cntt_2026"],"must_include":["Khoa học máy tính","Mạng","phần mềm"],"weakness_type":"multi_entity"},
    {"id":"q196","category":"multihop","question":"Ngành nào tại HUSC có thời gian đào tạo 5 năm?","ground_truth_answer":"Kiến trúc là ngành duy nhất đào tạo 5 năm tại HUSC","gt_chunks":["tuyensinh_kientruc_2026"],"must_include":["Kiến trúc","5 năm"],"weakness_type":"multi_entity"},
    {"id":"q197","category":"multihop","question":"Liệt kê tất cả ngành thuộc khối ngành V tại HUSC và học phí.","ground_truth_answer":"CNTT, KTPM, KHMT, Tin học, Kiến trúc. Học phí ~600000đ/tín chỉ","gt_chunks":["tuyensinh_overview_2026"],"must_include":["CNTT","Kiến trúc","600000"],"weakness_type":"multi_entity"},
    {"id":"q198","category":"multihop","question":"So sánh tổng chi phí học 4 năm giữa ngành Văn học và CNTT.","ground_truth_answer":"Văn học: ~525000x120=63000000đ, CNTT: ~600000x120=72000000đ. Chênh 9 triệu đồng","gt_chunks":["tuyensinh_vanhoc_2026","tuyensinh_cntt_2026"],"must_include":["525000","600000"],"weakness_type":"multi_entity"},

    # ==========================================
    # CATEGORY 3: ENUMERATION COMPLETENESS (25 questions)
    # ==========================================
    {"id":"q199","category":"multihop","question":"Liệt kê TẤT CẢ 26 ngành đào tạo của HUSC năm 2026 theo chỉ tiêu từ cao xuống thấp.","ground_truth_answer":"1.CNTT(510), 2.KTPM, 3.KHMT, 4.Tin học(120), 5.Toán(100), 6.Hóa(90), 7.Lý(80), 8.Sinh(70), 9.Văn(60), 10.Báo chí(60), 11.Luật(60), 12.Kiến trúc(60), 13.Sư phạm Toán, 14.Sư phạm Tin học, 15.Địa lý(50), 16.Lịch sử(50), 17.Triết, 18.XH học, 19.Đông phương, 20.Truyền thông, 21.VN học, 22.QL công, 23.Chính trị, 24.GD thể chất, 25.Khoa học dữ liệu, 26.Quản lý KT","gt_chunks":["tuyensinh_overview_2026"],"must_include":["26","Công nghệ thông tin","510"],"weakness_type":"enumeration"},
    {"id":"q200","category":"multihop","question":"Liệt kê tất cả tổ hợp xét tuyển mà HUSC sử dụng năm 2026.","ground_truth_answer":"A00, A01, B00, C00, C01, C02, C19, D01, D07, D14, V00, V01, V02, V12, X06, X70 và một số tổ hợp khác","gt_chunks":["tuyensinh_overview_2026"],"must_include":["A00","C00","V00","tổ hợp"],"weakness_type":"enumeration"},
    {"id":"q201","category":"multihop","question":"Liệt kê tất cả phương thức xét tuyển tại HUSC năm 2026.","ground_truth_answer":"1) Thi THPT (mã 100), 2) Xét học bạ THPT (mã 200), 3) Xét tuyển thẳng (mã 301), 4) Đánh giá năng lực, 5) Thi năng khiếu","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["100","200","301","năng lực"],"weakness_type":"enumeration"},
    {"id":"q202","category":"multihop","question":"Liệt kê tất cả các trường mà HUSC chấp nhận kết quả thi Vẽ mỹ thuật.","ground_truth_answer":"ĐH Kiến trúc Hà Nội, ĐH Xây dựng Hà Nội, ĐH Bách khoa Đà Nẵng, ĐH Kiến trúc TP.HCM, ĐH Bách khoa ĐHQG TP.HCM","gt_chunks":["tuyensinh_kientruc_2026"],"must_include":["Kiến trúc","Xây dựng","Bách khoa"],"weakness_type":"enumeration"},
    {"id":"q203","category":"multihop","question":"Liệt kê tất cả các mức điểm thưởng cho giải HSG/KHKT tại HUSC.","ground_truth_answer":"Giải Nhất: +3.0, Giải Nhì: +2.5, Giải Ba: +2.0, Giải Khuyến khích: +1.5","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["3.0","2.5","2.0","1.5"],"weakness_type":"enumeration"},
    {"id":"q204","category":"multihop","question":"Liệt kê tất cả mức ưu tiên khu vực và điểm cộng tương ứng.","ground_truth_answer":"KV1: +0.75, KV2-NT: +0.5, KV2: +0.25, KV3: +0","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["0.75","0.5","0.25"],"weakness_type":"enumeration"},
    {"id":"q205","category":"multihop","question":"Liệt kê tất cả chứng chỉ tiếng Anh được HUSC chấp nhận quy đổi điểm.","ground_truth_answer":"IELTS, TOEFL iBT, VSTEP","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["IELTS","TOEFL","VSTEP"],"weakness_type":"enumeration"},
    {"id":"q206","category":"multihop","question":"Liệt kê tất cả tổ hợp xét tuyển ngành CNTT HUSC 2026.","ground_truth_answer":"A00, A01, C01, D07","gt_chunks":["tuyensinh_cntt_2026"],"must_include":["A00","A01","C01","D07"],"weakness_type":"enumeration"},
    {"id":"q207","category":"multihop","question":"Liệt kê tất cả tổ hợp xét tuyển ngành Toán học HUSC 2026.","ground_truth_answer":"A00, A01, C01","gt_chunks":["tuyensinh_toan_2026"],"must_include":["A00","A01","C01"],"weakness_type":"enumeration"},
    {"id":"q208","category":"multihop","question":"Liệt kê tất cả chuyên ngành của ngành CNTT HUSC.","ground_truth_answer":"1) Khoa học máy tính (trọng tâm AI), 2) Mạng máy tính, 3) Công nghệ phần mềm","gt_chunks":["tuyensinh_cntt_2026"],"must_include":["Khoa học máy tính","Mạng máy tính","Công nghệ phần mềm"],"weakness_type":"enumeration"},
    {"id":"q209","category":"multihop","question":"Liệt kê tất cả các khoa của HUSC.","ground_truth_answer":"Khoa Toán, Khoa CNTT, Khoa Vật lý, Khoa Hóa học, Khoa Sinh học, Khoa Địa lý, Khoa Văn học, Khoa Lịch sử, Khoa Luật, Khoa Kiến trúc","gt_chunks":["husc_info"],"must_include":["Khoa Toán","Khoa CNTT","Khoa Vật lý","Khoa Hóa học"],"weakness_type":"enumeration"},
    {"id":"q210","category":"multihop","question":"Liệt kê tất cả hồ sơ cần nộp cho xét tuyển thẳng HUSC 2026.","ground_truth_answer":"1) Phiếu đăng ký xét tuyển thẳng (Phụ lục 2), 2) Bản photocopy học bạ THPT, 3) Minh chứng thành tích/giấy xác nhận đối tượng","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["Phụ lục 2","học bạ","minh chứng"],"weakness_type":"enumeration"},
    {"id":"q211","category":"multihop","question":"Liệt kê tất cả các mốc thời gian quan trọng trong tuyển sinh HUSC 2026.","ground_truth_answer":"Đăng ký NV đợt 1: 2/7-14/7/2026, Nộp lệ phí: 15/7-21/7/2026, Xác nhận nhập học: 22/8-17h00 30/8/2026","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["2/7","14/7","30/8"],"weakness_type":"enumeration"},
    {"id":"q212","category":"multihop","question":"Liệt kê tất cả các loại học bổng HUSC cung cấp cho sinh viên.","ground_truth_answer":"1) Học bổng điểm đầu vào (thủ khoa, >=25đ, 24-<25đ), 2) Học bổng kết quả học tập, 3) Học bổng từ doanh nghiệp/tổ chức","gt_chunks":["hocbong_2026"],"must_include":["thủ khoa","điểm đầu vào","doanh nghiệp"],"weakness_type":"enumeration"},
    {"id":"q213","category":"multihop","question":"Liệt kê tất cả các ngành có học phí miễn tại HUSC.","ground_truth_answer":"Chỉ có Triết học được miễn học phí hoàn toàn","gt_chunks":["tuyensinh_trieuthoc_2026"],"must_include":["Triết","miễn"],"weakness_type":"enumeration"},
    {"id":"q214","category":"multihop","question":"Liệt kê tất cả ngành xét tuyển bằng thi năng khiếu vẽ tại HUSC.","ground_truth_answer":"Chỉ có ngành Kiến trúc (tổ hợp V00, V01, V02, V12)","gt_chunks":["tuyensinh_kientruc_2026"],"must_include":["Kiến trúc","V00"],"weakness_type":"enumeration"},
    {"id":"q215","category":"multihop","question":"Liệt kê tất cả các đối tượng được xét tuyển thẳng vào HUSC 2026.","ground_truth_answer":"Anh hùng LĐ/LĐQ, Chiến sĩ thi đua toàn quốc, Giải HSG quốc gia/quốc tế, Người khuyết tật nặng, Dự bị đại học","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["Anh hùng","HSG","khuyết tật"],"weakness_type":"enumeration"},
    {"id":"q216","category":"multihop","question":"Liệt kê tất cả các ngành đào tạo thuộc khoa Toán HUSC.","ground_truth_answer":"Toán học và Tin học","gt_chunks":["tuyensinh_toan_2026","tuyensinh_tinhoc_2026"],"must_include":["Toán học","Tin học"],"weakness_type":"enumeration"},
    {"id":"q217","category":"multihop","question":"Liệt kê tất cả ngành có chỉ tiêu 60 tại HUSC 2026.","ground_truth_answer":"Văn học, Báo chí, Luật, Kiến trúc","gt_chunks":["tuyensinh_overview_2026"],"must_include":["Văn","Báo chí","Luật","Kiến trúc"],"weakness_type":"enumeration"},
    {"id":"q218","category":"multihop","question":"Liệt kê tất cả ngành có chỉ tiêu 50 tại HUSC 2026.","ground_truth_answer":"Địa lý và Lịch sử","gt_chunks":["tuyensinh_overview_2026"],"must_include":["Địa lý","Lịch sử","50"],"weakness_type":"enumeration"},
    {"id":"q219","category":"multihop","question":"Liệt kê tất cả các bước đăng ký xét tuyển vào HUSC năm 2026.","ground_truth_answer":"1) Đăng ký NV trên hệ thống Bộ GDĐT, 2) Nộp lệ phí, 3) Xử lý NV, 4) Xác nhận nhập học trực tuyến, 5) Nộp hồ sơ nhập học","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["đăng ký","xác nhận","nhập học"],"weakness_type":"enumeration"},
    {"id":"q220","category":"multihop","question":"Liệt kê tất cả ngành đào tạo sư phạm tại HUSC 2026.","ground_truth_answer":"Sư phạm Toán học, Sư phạm Tin học","gt_chunks":["tuyensinh_sptoan_2026","tuyensinh_sptinhoc_2026"],"must_include":["Sư phạm Toán","Sư phạm Tin"],"weakness_type":"enumeration"},
    {"id":"q221","category":"multihop","question":"Liệt kê tất cả tổ hợp xét tuyển ngành Kiến trúc HUSC.","ground_truth_answer":"V00, V01, V02, V12","gt_chunks":["tuyensinh_kientruc_2026"],"must_include":["V00","V01","V02","V12"],"weakness_type":"enumeration"},
    {"id":"q222","category":"multihop","question":"Liệt kê tất cả mức học bổng theo điểm đầu vào HUSC 2026.","ground_truth_answer":">=25đ: 100% học phí HK1, 24-<25đ: 50% học phí HK1, thủ khoa: 100% học phí HK1","gt_chunks":["hocbong_2026"],"must_include":["100%","50%","25"],"weakness_type":"enumeration"},
    {"id":"q223","category":"multihop","question":"Liệt kê tất cả các loại điểm được cộng vào điểm xét tuyển HUSC 2026.","ground_truth_answer":"Điểm ưu tiên khu vực (KV1, KV2-NT, KV2), Điểm ưu tiên đối tượng, Điểm thưởng (giải HSG/KHKT)","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["ưu tiên khu vực","ưu tiên đối tượng","điểm thưởng"],"weakness_type":"enumeration"},

    # ==========================================
    # CATEGORY 4: NEGATION/EXCLUSION (15 questions)
    # ==========================================
    {"id":"q224","category":"simple","question":"Trường có ngành Y học không?","ground_truth_answer":"Không, HUSC không có ngành Y học. HUSC chuyên về khoa học cơ bản và ứng dụng.","gt_chunks":[],"must_include":["không"],"weakness_type":"negation"},
    {"id":"q225","category":"simple","question":"Trường có ngành Kinh tế không?","ground_truth_answer":"Không, HUSC không đào tạo ngành Kinh tế. Đại học Huế có trường Kinh tế riêng.","gt_chunks":[],"must_include":["không"],"weakness_type":"negation"},
    {"id":"q226","category":"simple","question":"Trường có ngành Ngoại ngữ (Tiếng Anh) không?","ground_truth_answer":"Không, HUSC không có ngành Ngoại ngữ. Đại học Huế có trường Ngoại ngữ riêng.","gt_chunks":[],"must_include":["không"],"weakness_type":"negation"},
    {"id":"q227","category":"simple","question":"HUSC có xét tuyển bằng điểm SAT không?","ground_truth_answer":"Không có thông tin về xét tuyển bằng SAT trong tài liệu tuyển sinh HUSC 2026.","gt_chunks":[],"must_include":["không"],"weakness_type":"negation"},
    {"id":"q228","category":"simple","question":"Trường có đào tạo trình độ cao đẳng không?","ground_truth_answer":"Không, HUSC chỉ đào tạo trình độ đại học.","gt_chunks":[],"must_include":["không"],"weakness_type":"negation"},
    {"id":"q229","category":"simple","question":"HUSC có có cổng xét tuyển riêng không?","ground_truth_answer":"Không, HUSC không có cổng xét tuyển riêng. Thí sinh đăng ký trên hệ thống Bộ GDĐT.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["không","Bộ GDĐT"],"weakness_type":"negation"},
    {"id":"q230","category":"simple","question":"Trường có bắt buộc chọn HUSC là NV1 không?","ground_truth_answer":"Không, không có yêu cầu bắt buộc chọn HUSC là nguyện vọng 1.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["không","bắt buộc"],"weakness_type":"negation"},
    {"id":"q231","category":"simple","question":"Xét học bạ có nhân hệ số môn nào không?","ground_truth_answer":"Không, xét học bạ THPT không nhân hệ số các môn.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["không","nhân hệ số"],"weakness_type":"negation"},
    {"id":"q232","category":"simple","question":"Trường có xét tuyển bằng điểm học bạ cấp 3 không (chỉ lớp 10-12)?","ground_truth_answer":"Không, HUSC xét học bạ THPT dựa trên điểm trung bình 4 học kỳ lớp 11 và lớp 12, không chỉ cấp 3 nói chung.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["lớp 11","lớp 12"],"weakness_type":"negation"},
    {"id":"q233","category":"simple","question":"Đã xác nhận nhập học ở trường khác thì còn được xét HUSC nữa không?","ground_truth_answer":"Không, thí sinh đã xác nhận nhập học ở trường khác thì không được tham gia xét tuyển ở nơi khác.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["không"],"weakness_type":"negation"},
    {"id":"q234","category":"simple","question":"Trường có ngành Dược không?","ground_truth_answer":"Không, HUSC không đào tạo ngành Dược.","gt_chunks":[],"must_include":["không"],"weakness_type":"negation"},
    {"id":"q235","category":"simple","question":"HUSC có chương trình đào tạo từ xa không?","ground_truth_answer":"Không có thông tin về đào tạo từ xa trong tài liệu tuyển sinh.","gt_chunks":[],"must_include":["không"],"weakness_type":"negation"},
    {"id":"q236","category":"simple","question":"Trường có yêu cầu thi thể lực cho ngành GD thể chất không?","ground_truth_answer":"Không có thông tin chi tiết về yêu cầu thi thể lực trong tài liệu hiện có.","gt_chunks":[],"must_include":["không"],"weakness_type":"negation"},
    {"id":"q237","category":"simple","question":"HUSC có học hệ vừa làm vừa học không?","ground_truth_answer":"Không có thông tin về hệ vừa làm vừa học trong tài liệu tuyển sinh HUSC 2026.","gt_chunks":[],"must_include":["không"],"weakness_type":"negation"},
    {"id":"q238","category":"simple","question":"Trường có đào tạo thạc sĩ ngành CNTT không?","ground_truth_answer":"Thông tin tuyển sinh hiện tại chỉ về đại học chính quy, không có thông tin về thạc sĩ.","gt_chunks":[],"must_include":["không"],"weakness_type":"negation"},

    # ==========================================
    # CATEGORY 5: CONTEXT CONFLICT (15 questions)
    # ==========================================
    {"id":"q239","category":"temporal","question":"Học phí CNTT năm 2024 là bao nhiêu?","ground_truth_answer":"Dữ liệu 2024 đã cũ, không còn chính xác. Tham khảo học phí 2026.","gt_chunks":[],"must_include":["không","2024"],"weakness_type":"context_conflict"},
    {"id":"q240","category":"temporal","question":"Điểm chuẩn ngành Toán năm 2023 là bao nhiêu?","ground_truth_answer":"Dữ liệu 2023 không còn trong hệ thống. Tham khảo điểm chuẩn 2026.","gt_chunks":[],"must_include":["không","2023"],"weakness_type":"context_conflict"},
    {"id":"q241","category":"temporal","question":"Chỉ tiêu ngành CNTT năm 2024 có bao nhiêu?","ground_truth_answer":"Dữ liệu 2024 đã cũ. Tham khảo chỉ tiêu 2026.","gt_chunks":[],"must_include":["không","2024"],"weakness_type":"context_conflict"},
    {"id":"q242","category":"temporal","question":"Học phí năm 2027 sẽ là bao nhiêu?","ground_truth_answer":"Chưa có thông tin học phí 2027. Học phí 2026 là thông tin mới nhất.","gt_chunks":[],"must_include":["không","2027"],"weakness_type":"context_conflict"},
    {"id":"q243","category":"temporal","question":"Thông tin tuyển sinh năm 2023 còn áp dụng được không?","ground_truth_answer":"Không, thông tin 2023 đã quá cũ. Chỉ dùng thông tin tuyển sinh 2026.","gt_chunks":[],"must_include":["không","cũ"],"weakness_type":"context_conflict"},
    {"id":"q244","category":"temporal","question":"Học phí 2025 là 545000/tín chỉ hay khác?","ground_truth_answer":"Năm 2025 học phí theo khối ngành: IV 505000, V 545000-580000, VII 440000. Khác với năm 2026.","gt_chunks":["tuyensinh_overview_2025"],"must_include":["505000","440000"],"weakness_type":"context_conflict"},
    {"id":"q245","category":"temporal","question":"Điểm chuẩn năm 2025 và 2026 cái nào cao hơn?","ground_truth_answer":"Điểm chuẩn 2026 thường cao hơn hoặc tương đương 2025 tùy ngành.","gt_chunks":["tuyensinh_overview_2025","tuyensinh_overview_2026"],"must_include":["2025","2026"],"weakness_type":"context_conflict"},
    {"id":"q246","category":"temporal","question":"Học phí khối V năm 2025 là 545000 hay 600000?","ground_truth_answer":"Năm 2025 khối V: 545000-580000đ/tín chỉ. Năm 2026 khối V: ~600000đ/tín chỉ. Hai năm khác nhau.","gt_chunks":["tuyensinh_overview_2025","tuyensinh_overview_2026"],"must_include":["545000","600000"],"weakness_type":"context_conflict"},
    {"id":"q247","category":"temporal","question":"Chỉ tiêu năm 2025 có giống 2026 không?","ground_truth_answer":"Không hoàn toàn giống, chỉ tiêu có thể thay đổi giữa các năm.","gt_chunks":["tuyensinh_overview_2025","tuyensinh_overview_2026"],"must_include":["thay đổi","không"],"weakness_type":"context_conflict"},
    {"id":"q248","category":"temporal","question":"Phương thức xét tuyển 2025 có khác 2026 không?","ground_truth_answer":"Về cơ bản giữ nguyên, có điều chỉnh theo Thông tư 06/2026.","gt_chunks":["dieukien_tuyensinh_2025","dieukien_tuyensinh_2026"],"must_include":["Thông tư","2026"],"weakness_type":"context_conflict"},
    {"id":"q249","category":"temporal","question":"Học phí HUSC 2026 có thể thay đổi không?","ground_truth_answer":"Học phí 2026 đã được công bố theo khối ngành nhưng có thể được điều chỉnh.","gt_chunks":["tuyensinh_overview_2026"],"must_include":["điều chỉnh","hoặc","thay đổi"],"weakness_type":"context_conflict"},
    {"id":"q250","category":"temporal","question":"Điểm chuẩn 2026 có thể tăng hay giảm so với khi công bố?","ground_truth_answer":"Điểm chuẩn được Hội đồng tuyển sinh quyết định dựa trên chỉ tiêu và chất lượng hồ sơ, có thể khác dự kiến.","gt_chunks":["tuyensinh_overview_2026"],"must_include":["Hội đồng","chỉ tiêu"],"weakness_type":"context_conflict"},
    {"id":"q251","category":"temporal","question":"Năm 2025 tổ hợp xét tuyển có khác 2026 không?","ground_truth_answer":"Về cơ bản tổ hợp xét tuyển giữ nguyên, có thể có điều chỉnh nhỏ theo Thông tư mới.","gt_chunks":["dieukien_tuyensinh_2025","dieukien_tuyensinh_2026"],"must_include":["tổ hợp","điều chỉnh"],"weakness_type":"context_conflict"},
    {"id":"q252","category":"temporal","question":"Học bổng 2025 và 2026 có gì khác nhau?","ground_truth_answer":"Chính sách học bổng cơ bản giữ nguyên, mức hỗ trợ có thể điều chỉnh theo học phí.","gt_chunks":["hocbong_2026"],"must_include":["học bổng"],"weakness_type":"context_conflict"},
    {"id":"q253","category":"temporal","question":"Có thông tin tuyển sinh bổ sung đợt 2 năm 2026 chưa?","ground_truth_answer":"Thông tin tuyển sinh bổ sung thường được công bố sau đợt 1. Chưa có thông tin cụ thể.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["chưa","bổ sung"],"weakness_type":"context_conflict"},

    # ==========================================
    # CATEGORY 6: CONVERSATIONAL/IMPRECISE (30 questions)
    # ==========================================
    {"id":"q254","category":"simple","question":"Ad ơi trường mình nộp hồ sơ ở đâu vậy","ground_truth_answer":"Nộp hồ sơ trực tiếp hoặc chuyển phát nhanh về: Ban Đào tạo và CTSV, số 01 Điện Biên Phủ, phường Vĩnh Ninh, TP. Huế. Hoặc đăng ký online trên hệ thống Bộ GDĐT.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["Điện Biên Phủ","Bộ GDĐT"],"weakness_type":"conversational"},
    {"id":"q255","category":"simple","question":"Cho e hỏi xét học bạ cần những gì ạ","ground_truth_answer":"Cần: học bạ THPT 3 năm, điểm trung bình 4 học kỳ lớp 11-12 (mức khá trở lên), đăng ký trên hệ thống Bộ GDĐT.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["học bạ","THPT"],"weakness_type":"conversational"},
    {"id":"q256","category":"simple","question":"Trường ơi năm nay điểm có tăng không ad","ground_truth_answer":"Chưa có dự báo chính thức. Điểm chuẩn phụ thuộc vào chỉ tiêu và chất lượng hồ sơ đăng ký.","gt_chunks":["tuyensinh_overview_2026"],"must_include":["điểm chuẩn"],"weakness_type":"conversational"},
    {"id":"q257","category":"multihop","question":"e muốn hỏi ngành nào hot nhất trường mình","ground_truth_answer":"CNTT là ngành hot nhất với chỉ tiêu 510 (cao nhất), điểm chuẩn cũng cao nhất.","gt_chunks":["tuyensinh_cntt_2026"],"must_include":["CNTT","510"],"weakness_type":"conversational"},
    {"id":"q258","category":"simple","question":"ĐK nguyện vọng ở đâu vậy ad","ground_truth_answer":"Đăng ký trên Hệ thống của Bộ GDĐT: thisinh.thitotnghiepthpt.edu.vn hoặc Cổng dịch vụ công quốc gia.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["Bộ GDĐT","đăng ký"],"weakness_type":"conversational"},
    {"id":"q259","category":"simple","question":"Trường có xét học bạ k ad","ground_truth_answer":"Có, HUSC xét tuyển học bạ THPT (mã 200).","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["Có","học bạ"],"weakness_type":"conversational"},
    {"id":"q260","category":"simple","question":"hoc phi khoa cntt bnh nhieu","ground_truth_answer":"Học phí CNTT khoảng 600000 VNĐ/tín chỉ (khối ngành V).","gt_chunks":["tuyensinh_cntt_2026"],"must_include":["600000"],"weakness_type":"conversational"},
    {"id":"q261","category":"simple","question":"mã trường husc là j","ground_truth_answer":"Mã trường HUSC là DHT.","gt_chunks":["husc_info"],"must_include":["DHT"],"weakness_type":"conversational"},
    {"id":"q262","category":"multihop","question":"ngành nào dễ đậu nhất trường mình","ground_truth_answer":"Các ngành có điểm chuẩn thấp nhất thường dễ đậu nhất: Vật lý (~16.5), Sinh học (~17.5).","gt_chunks":["tuyensinh_overview_2026"],"must_include":["Vật lý","Sinh học"],"weakness_type":"conversational"},
    {"id":"q263","category":"simple","question":"ktx trường có k ad","ground_truth_answer":"Có, HUSC có ký túc xá. Đăng ký online tại http://sv-dhh.kytucxa.vn/","gt_chunks":["husc_info"],"must_include":["Có","ký túc xá"],"weakness_type":"conversational"},
    {"id":"q264","category":"multihop","question":"cho e hỏi học phí trường mình rẻ hay đắt","ground_truth_answer":"Học phí HUSC ở mức trung bình: 525000-600000đ/tín chỉ tùy khối ngành. So với các trường khác cùng khu vực thì vừa phải.","gt_chunks":["tuyensinh_overview_2026"],"must_include":["525000","600000"],"weakness_type":"conversational"},
    {"id":"q265","category":"simple","question":"hạn chót nộp hồ sơ khi nào ad ơi","ground_truth_answer":"Hạn nộp lệ phí: đến 17h00 ngày 21/7/2026. Hạn xác nhận nhập học: đến 17h00 ngày 30/8/2026.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["17h00","2026"],"weakness_type":"conversational"},
    {"id":"q266","category":"simple","question":"trường ở đâu vậy","ground_truth_answer":"HUSC ở 77 Nguyễn Huệ, phường Phú Nhuận, TP. Huế","gt_chunks":["husc_info"],"must_include":["Nguyễn Huệ","Huế"],"weakness_type":"conversational"},
    {"id":"q267","category":"simple","question":"sđt liên hệ tuyển sinh","ground_truth_answer":"094 4455136","gt_chunks":["husc_info"],"must_include":["094"],"weakness_type":"conversational"},
    {"id":"q268","category":"multihop","question":"xét học bạ cần điểm bao nhiêu mới đậu","ground_truth_answer":"Tùy ngành. Điều kiện chung: kết quả TB học tập mức khá trở lên, điểm tổ hợp 3 môn từ 15.00 trở lên (thang 30).","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["khá","15"],"weakness_type":"conversational"},
    {"id":"q269","category":"simple","question":"có được xét 2 ngành trong trường k","ground_truth_answer":"Có, thí sinh được đăng ký tối đa 15 nguyện vọng, có thể chọn nhiều ngành trong cùng trường.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["15","nguyện vọng"],"weakness_type":"conversational"},
    {"id":"q270","category":"simple","question":"xét thẳng cần gì ạ","ground_truth_answer":"Cần: Phiếu đăng ký xét tuyển thẳng (Phụ lục 2), học bạ THPT, minh chứng thành tích/đối tượng.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["Phụ lục 2","minh chứng"],"weakness_type":"conversational"},
    {"id":"q271","category":"multihop","question":"ngành nào có việc làm nhiều nhất","ground_truth_answer":"CNTT có cơ hội việc làm nhiều nhất do nhu cầu thị trường cao, chỉ tiêu tuyển sinh cũng cao nhất (510).","gt_chunks":["tuyensinh_cntt_2026"],"must_include":["CNTT"],"weakness_type":"conversational"},
    {"id":"q272","category":"simple","question":"website trường là gì","ground_truth_answer":"https://husc.edu.vn","gt_chunks":["husc_info"],"must_include":["husc.edu.vn"],"weakness_type":"conversational"},
    {"id":"q273","category":"simple","question":"nộp hồ sơ online hay offline","ground_truth_answer":"Đăng ký xét tuyển online trên hệ thống Bộ GDĐT. Sau khi trúng tuyển, nộp hồ sơ giấy về trường hoặc chuyển phát nhanh.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["online","Bộ GDĐT"],"weakness_type":"conversational"},
    {"id":"q274","category":"multihop","question":"cho e hỏi ngành kiến trúc thi môn gì","ground_truth_answer":"Thi Vẽ mỹ thuật (năng khiếu), phải đạt từ 5.0 trở lên. Tổ hợp: V00, V01, V02, V12.","gt_chunks":["tuyensinh_kientruc_2026"],"must_include":["Vẽ mỹ thuật","5.0"],"weakness_type":"conversational"},
    {"id":"q275","category":"simple","question":"em khuyết tật có được xét thẳng không","ground_truth_answer":"Có, người khuyết tật nặng được xét tuyển thẳng. Cần nộp Giấy xác nhận khuyết tật nặng.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["khuyết tật","xét tuyển thẳng"],"weakness_type":"conversational"},
    {"id":"q276","category":"multihop","question":"chuyên ngành AI học ở trường nào","ground_truth_answer":"Chuyên ngành Khoa học máy tính (AI) thuộc ngành CNTT tại HUSC. Mã ngành CNTT: 7480201.","gt_chunks":["tuyensinh_cntt_2026"],"must_include":["CNTT","AI"],"weakness_type":"conversational"},
    {"id":"q277","category":"simple","question":"thi xong r thì đăng ký NV kiểu gì","ground_truth_answer":"Đăng ký nguyện vọng trên Hệ thống Bộ GDĐT: thisinh.thitotnghiepthpt.edu.vn, tối đa 15 nguyện vọng.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["Bộ GDĐT","15"],"weakness_type":"conversational"},
    {"id":"q278","category":"multihop","question":"có cần gửi hồ sơ giấy về trường k","ground_truth_answer":"Có, sau khi trúng tuyển phải nộp hồ sơ giấy trực tiếp hoặc chuyển phát nhanh về: 01 Điện Biên Phủ, Vĩnh Ninh, Huế.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["hồ sơ","Điện Biên Phủ"],"weakness_type":"conversational"},
    {"id":"q279","category":"simple","question":"điểm ưu tiên KV1 cộng bnh nhieu","ground_truth_answer":"KV1 được cộng 0.75 điểm ưu tiên khu vực.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["0.75"],"weakness_type":"conversational"},
    {"id":"q280","category":"simple","question":"học bạ 3 năm đều khá được cộng điểm k","ground_truth_answer":"Hạnh kiểm Khá vẫn được xét. Điểm thưởng cộng cho giải HSG, không phải hạnh kiểm. 3 năm HS Khá không được cộng điểm thưởng.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["không","giải"],"weakness_type":"conversational"},
    {"id":"q281","category":"multihop","question":"học 2 ngành cùng lúc được k","ground_truth_answer":"Thông tin về học song ngành không có trong tài liệu tuyển sinh. Cần liên hệ Phòng Đào tạo.","gt_chunks":[],"must_include":["không","liên hệ"],"weakness_type":"conversational"},
    {"id":"q282","category":"simple","question":"IELTS 6.0 quy đổi bnh nhieu điểm","ground_truth_answer":"IELTS 6.0 quy đổi khoảng 8.0-9.0 điểm thi THPT môn Tiếng Anh","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["IELTS","8"],"weakness_type":"conversational"},
    {"id":"q283","category":"simple","question":"nhà ở xa có được ưu tiên gì k","ground_truth_answer":"Có, thí sinh vùng sâu vùng xa thuộc KV1 được cộng 0.75 điểm, KV2-NT cộng 0.5 điểm ưu tiên khu vực.","gt_chunks":["dieukien_tuyensinh_2026"],"must_include":["KV1","0.75"],"weakness_type":"conversational"},

    # ==========================================
    # CATEGORY 7: CROSS-YEAR COMPARISON (20 questions)
    # ==========================================
    {"id":"q284","category":"temporal","question":"So sánh chỉ tiêu ngành CNTT 2025 và 2026.","ground_truth_answer":"Chỉ tiêu CNTT 2026 là 510. Năm 2025 có thể khác, thường giữ ổn định hoặc tăng nhẹ.","gt_chunks":["tuyensinh_cntt_2025","tuyensinh_cntt_2026"],"must_include":["2025","2026","510"],"weakness_type":"cross_year"},
    {"id":"q285","category":"temporal","question":"Học phí tăng bao nhiêu % từ 2025 sang 2026?","ground_truth_answer":"Khối VII: 440000→525000 (+19%), khối IV: 505000→540000 (+7%), khối V: 545000→600000 (+10%)","gt_chunks":["tuyensinh_overview_2025","tuyensinh_overview_2026"],"must_include":["440000","525000"],"weakness_type":"cross_year"},
    {"id":"q286","category":"temporal","question":"So sánh điểm chuẩn ngành Toán 2025 và 2026.","ground_truth_answer":"Điểm chuẩn Toán 2026 là 18.5. So với 2025 có thể tăng nhẹ.","gt_chunks":["tuyensinh_toan_2025","tuyensinh_toan_2026"],"must_include":["18.5","2025","2026"],"weakness_type":"cross_year"},
    {"id":"q287","category":"temporal","question":"Học phí khối IV năm 2025 so với 2026 thay đổi thế nào?","ground_truth_answer":"2025: 505000đ/tín chỉ, 2026: ~540000đ/tín chỉ. Tăng 35000đ/tín chỉ (~7%).","gt_chunks":["tuyensinh_overview_2025","tuyensinh_overview_2026"],"must_include":["505000","540000"],"weakness_type":"cross_year"},
    {"id":"q288","category":"temporal","question":"So sánh học phí khối VII giữa năm 2025 và 2026.","ground_truth_answer":"2025: 440000đ/tín chỉ, 2026: ~525000đ/tín chỉ. Tăng 85000đ/tín chỉ (~19%).","gt_chunks":["tuyensinh_overview_2025","tuyensinh_overview_2026"],"must_include":["440000","525000"],"weakness_type":"cross_year"},
    {"id":"q289","category":"temporal","question":"Học phí khối V tăng từ bao nhiêu lên bao nhiêu từ 2025 sang 2026?","ground_truth_answer":"Từ 545000-580000đ/tín chỉ (2025) lên ~600000đ/tín chỉ (2026).","gt_chunks":["tuyensinh_overview_2025","tuyensinh_overview_2026"],"must_include":["545000","600000"],"weakness_type":"cross_year"},
    {"id":"q290","category":"temporal","question":"Chỉ tiêu ngành Hóa học 2025 so với 2026 có khác không?","ground_truth_answer":"Chỉ tiêu Hóa học 2026 là 90. Năm 2025 có thể tương đương.","gt_chunks":["tuyensinh_hoahoc_2025","tuyensinh_hoahoc_2026"],"must_include":["90","2025","2026"],"weakness_type":"cross_year"},
    {"id":"q291","category":"temporal","question":"So sánh tổ hợp xét tuyển ngành CNTT 2025 vs 2026.","ground_truth_answer":"Tổ hợp xét tuyển CNTT về cơ bản giữ nguyên: A00, A01, C01, D07.","gt_chunks":["tuyensinh_cntt_2025","tuyensinh_cntt_2026"],"must_include":["A00","C01","2025","2026"],"weakness_type":"cross_year"},
    {"id":"q292","category":"temporal","question":"Điểm chuẩn Vật lý 2025 so với 2026 thay đổi thế nào?","ground_truth_answer":"Vật lý 2026 ~16.5 điểm. 2025 có thể tương đương hoặc thấp hơn một chút.","gt_chunks":["tuyensinh_vatly_2025","tuyensinh_vatly_2026"],"must_include":["2025","2026","Vật lý"],"weakness_type":"cross_year"},
    {"id":"q293","category":"temporal","question":"Tổng chỉ tiêu HUSC 2025 so với 2026 tăng hay giảm?","ground_truth_answer":"Tổng chỉ tiêu có thể tăng nhẹ hoặc giữ ổn định. Năm 2026 tổng chỉ tiêu khoảng 460+.","gt_chunks":["tuyensinh_overview_2025","tuyensinh_overview_2026"],"must_include":["chỉ tiêu","2025","2026"],"weakness_type":"cross_year"},
    {"id":"q294","category":"temporal","question":"Thông tư tuyển sinh 2025 và 2026 khác nhau điểm gì?","ground_truth_answer":"Năm 2026 áp dụng Thông tư 06/2026/TT-BGDĐT với một số điều chỉnh về phương thức và điều kiện tuyển sinh.","gt_chunks":["thongtu_06_2026"],"must_include":["Thông tư","2026"],"weakness_type":"cross_year"},
    {"id":"q295","category":"temporal","question":"Học phí 2024 so với 2026 thay đổi bao nhiêu?","ground_truth_answer":"Học phí tăng đáng kể từ 2024 sang 2026. 2024: khối IV ~460000, 2026: khối IV ~540000.","gt_chunks":["tuyensinh_overview_2026"],"must_include":["tăng","2026"],"weakness_type":"cross_year"},
    {"id":"q296","category":"temporal","question":"Ngành nào điểm chuẩn tăng nhiều nhất từ 2025 sang 2026?","ground_truth_answer":"Các ngành hot như CNTT, KTPM có thể tăng điểm chuẩn mạnh nhất.","gt_chunks":["tuyensinh_overview_2025","tuyensinh_overview_2026"],"must_include":["CNTT","tăng"],"weakness_type":"cross_year"},
    {"id":"q297","category":"temporal","question":"Chính sách điểm thưởng 2025 khác 2026 không?","ground_truth_answer":"Chính sách điểm thưởng cơ bản giữ nguyên: Nhất +3.0, Nhì +2.5, Ba +2.0, KK +1.5.","gt_chunks":["dieukien_tuyensinh_2025","dieukien_tuyensinh_2026"],"must_include":["3.0","2.5","2.0"],"weakness_type":"cross_year"},
    {"id":"q298","category":"temporal","question":"Hạn xác nhận nhập học 2025 so với 2026 có khác không?","ground_truth_answer":"2025: đến 17h00 30/08/2025. 2026: đến 17h00 21/8/2026 hoặc 30/8/2026.","gt_chunks":["dieukien_tuyensinh_2025","dieukien_tuyensinh_2026"],"must_include":["17h00","2025","2026"],"weakness_type":"cross_year"},
    {"id":"q299","category":"temporal","question":"Phương thức xét tuyển thẳng 2025 và 2026 có gì khác?","ground_truth_answer":"Về cơ bản giữ nguyên, chỉ tiêu tối đa 10% cho mỗi ngành.","gt_chunks":["dieukien_tuyensinh_2025","dieukien_tuyensinh_2026"],"must_include":["10%","xét tuyển thẳng"],"weakness_type":"cross_year"},
    {"id":"q300","category":"temporal","question":"So sánh học phí ngành Kiến trúc 2025 và 2026.","ground_truth_answer":"Kiến trúc thuộc khối V. 2025: 545000-580000đ/tín chỉ, 2026: ~600000đ/tín chỉ.","gt_chunks":["tuyensinh_overview_2025","tuyensinh_overview_2026"],"must_include":["545000","600000"],"weakness_type":"cross_year"},
    {"id":"q301","category":"temporal","question":"Điểm chuẩn sàn 2025 so với 2026 có tăng không?","ground_truth_answer":"Điểm sàn (mức thấp nhất) dao động khoảng 15.5-16.0, có thể tăng nhẹ vào 2026.","gt_chunks":["tuyensinh_overview_2025","tuyensinh_overview_2026"],"must_include":["điểm","2025","2026"],"weakness_type":"cross_year"},
    {"id":"q302","category":"temporal","question":"Triết học miễn học phí từ năm nào?","ground_truth_answer":"Triết học được miễn học phí theo chính sách nhà nước cho ngành sư phạm, áp dụng cả 2025 và 2026.","gt_chunks":["tuyensinh_trieuthoc_2026"],"must_include":["Triết","miễn"],"weakness_type":"cross_year"},
    {"id":"q303","category":"temporal","question":"Xét học bạ 2025 và 2026 có khác cách tính không?","ground_truth_answer":"Cách tính cơ bản giữ nguyên: điểm TB 4 học kỳ lớp 11-12, không nhân hệ số, cộng điểm ưu tiên.","gt_chunks":["dieukien_tuyensinh_2025","dieukien_tuyensinh_2026"],"must_include":["4 học kỳ","không nhân hệ số"],"weakness_type":"cross_year"},
]


def main():
    with TEST_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    # Add weakness_type to existing questions
    for q in questions:
        if "weakness_type" not in q:
            q["weakness_type"] = EXISTING_WEAKNESS_MAP.get(q["category"], "unknown")

    # Add new questions
    existing_ids = {q["id"] for q in questions}
    added = 0
    for nq in NEW_QUESTIONS:
        if nq["id"] not in existing_ids:
            questions.append(nq)
            added += 1

    # Sort by id number
    def sort_key(q):
        num = int(q["id"].lstrip("q"))
        return num

    questions.sort(key=sort_key)

    # Write back
    with TEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    # Stats
    total = len(questions)
    by_cat = {}
    by_weak = {}
    for q in questions:
        cat = q["category"]
        weak = q.get("weakness_type", "unknown")
        by_cat[cat] = by_cat.get(cat, 0) + 1
        by_weak[weak] = by_weak.get(weak, 0) + 1

    print(f"Added {added} new questions. Total: {total}")
    print(f"\nBy category:")
    for k, v in sorted(by_cat.items()):
        print(f"  {k}: {v}")
    print(f"\nBy weakness type:")
    for k, v in sorted(by_weak.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
