"""
Supplement 86Q test file with real student questions from admissions group chat.

- Fixes q017: strengthens must_include for enumeration testing
- Adds 37 real-world questions (q087-q123) covering:
  * 4 previously failed questions (gnd=0 in report.md)
  * Enumeration/listing questions (test GraphRAG)
  * Real student phrasing (natural language, multi-intent)
  * Admissions process questions
  * Comparison and temporal questions

Usage:
    cd rag2025
    python scripts/supplement_test.py
"""
from __future__ import annotations

import json
from pathlib import Path

TEST_PATH = Path(__file__).parent.parent / "data" / "eval" / "husc_86q_test.json"


# Real student questions from .omc/report.md (admissions group chat)
NEW_QUESTIONS = [
    {
        "id": "q087",
        "category": "multihop",
        "question": "Trường ĐH Khoa Học có bao nhiêu ngành đào tạo? Liệt kê tất cả các ngành theo thứ tự chỉ tiêu từ cao xuống thấp.",
        "ground_truth_answer": "26 ngành. CNTT 510 chỉ tiêu (cao nhất), tiếp theo KTPM, KHMT, Tin học 120, Toán học 100, Hóa học 90, Vật lý 80, Sinh học 70, Văn học 60, Lịch sử 50, Địa lý 50, Báo chí, Luật, Kiến trúc, Triết học, Xã hội học, Đông phương học, Sư phạm Toán, Sư phạm Tin học, Truyền thông số, Việt Nam học, Quản lý công, Chính trị học, Giáo dục thể chất, Khoa học máy tính.",
        "gt_chunks": ["tuyensinh_overview_2026"],
        "must_include": ["26", "Công nghệ thông tin", "510", "chỉ tiêu"],
    },
    {
        "id": "q088",
        "category": "multihop",
        "question": "Cách tính điểm học bạ của trường mình tính như thế nào ạ?",
        "ground_truth_answer": "Điểm xét tuyển học bạ = điểm trung bình 4 học kỳ (lớp 11 và lớp 12), làm tròn đến 1 chữ số thập phân. Công thức: Điểm xét tuyển = M1 + M2 + M3 + Điểm ưu tiên + Điểm thưởng.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["điểm trung bình", "4 học kỳ", "lớp 11", "lớp 12"],
    },
    {
        "id": "q089",
        "category": "multihop",
        "question": "Cách tính điểm xét tuyển ngành Kiến trúc?",
        "ground_truth_answer": "Ngành Kiến trúc xét tuyển tổ hợp V00/V01/V02/V12, bao gồm môn Vẽ mỹ thuật (phải đạt từ 5.0 điểm trở lên). Điểm xét tuyển = tổng điểm các môn trong tổ hợp + điểm ưu tiên + điểm thưởng.",
        "gt_chunks": ["tuyensinh_kientruc_2026"],
        "must_include": ["Vẽ mỹ thuật", "5.0", "V00"],
    },
    {
        "id": "q090",
        "category": "multihop",
        "question": "Điểm cộng, điểm ưu tiên này tính như thế nào ạ?",
        "ground_truth_answer": "Điểm ưu tiên khu vực: KV1 +0.75, KV2-NT +0.5, KV2 +0.25, KV3 +0. Điểm ưu tiên đối tượng theo quy định Bộ GDĐT. Điểm thưởng cho giải HSG/KHKT: Nhất +3.0, Nhì +2.5, Ba +2.0, KK +1.5.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["ưu tiên khu vực", "KV1", "0.75"],
    },
    {
        "id": "q091",
        "category": "multihop",
        "question": "Điểm ưu tiên và điểm thưởng tính như thế nào vậy ạ?",
        "ground_truth_answer": "Điểm ưu tiên khu vực: KV1 +0.75, KV2-NT +0.5, KV2 +0.25. Điểm thưởng: giải Nhất +3.0, Nhì +2.5, Ba +2.0, Khuyến khích +1.5 (chỉ cộng nếu không sử dụng quyền xét tuyển thẳng).",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["ưu tiên khu vực", "điểm thưởng", "3.0"],
    },
    {
        "id": "q092",
        "category": "multihop",
        "question": "Cho em hỏi 2 ngành CNTT và CNNT (Việt-Nhật) này khác gì nhau vậy ạ?",
        "ground_truth_answer": "CNTT (7480201) đào tạo 3 chuyên ngành: KHMT(AI), Mạng máy tính, Công nghệ phần mềm, chỉ tiêu 510. CNNT Việt-Nhật là chương trình liên kết quốc tế, học phí cao hơn, có thời gian học tại Nhật.",
        "gt_chunks": ["tuyensinh_cntt_2026", "tuyensinh_cnnt_2026"],
        "must_include": ["CNTT", "Việt-Nhật", "chuyên ngành"],
    },
    {
        "id": "q093",
        "category": "multihop",
        "question": "Ngành Kiến trúc của trường xét những tổ hợp nào ạ? Môn thi năng khiếu em ở xa, có thể thi ở các trường gần địa phương em được không hay phải về thi trực tiếp tại trường Đại học Khoa học?",
        "ground_truth_answer": "Ngành Kiến trúc xét tổ hợp V00, V01, V02, V12. Có thể dùng kết quả thi Vẽ mỹ thuật từ các trường khác: ĐH Kiến trúc Hà Nội, ĐH Xây dựng Hà Nội, ĐH Bách khoa Đà Nẵng, ĐH Kiến trúc TP.HCM. Điểm vẽ phải từ 5.0 trở lên.",
        "gt_chunks": ["tuyensinh_kientruc_2026"],
        "must_include": ["V00", "Vẽ mỹ thuật", "5.0"],
    },
    {
        "id": "q094",
        "category": "multihop",
        "question": "Cùng 1 ngành học thì mình được phép xét 2 phương thức k ạ. Dùng điểm thi tốt nghiệp và học bạ",
        "ground_truth_answer": "Có, thí sinh hoàn toàn có thể đăng ký một ngành với nhiều phương thức xét tuyển khác nhau (mã 100: thi THPT, mã 200: học bạ THPT). Điểm xét tuyển tính riêng cho mỗi phương thức.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["phương thức", "100", "200"],
    },
    {
        "id": "q095",
        "category": "multihop",
        "question": "Khối C01 20 điểm có thể đăng ký ngành nào của trường năm nay ạ?",
        "ground_truth_answer": "Với 20 điểm khối C01, có thể đăng ký: Vật lý học (tổ hợp C01), Toán học (C01), và một số ngành khác có tổ hợp C01.",
        "gt_chunks": ["tuyensinh_overview_2026"],
        "must_include": ["C01", "Vật lý"],
    },
    {
        "id": "q096",
        "category": "simple",
        "question": "Trường cho em xin mã trường của trường mình vs ạ",
        "ground_truth_answer": "Mã trường Đại học Khoa học - ĐH Huế là DHT.",
        "gt_chunks": ["husc_info"],
        "must_include": ["DHT"],
    },
    {
        "id": "q097",
        "category": "simple",
        "question": "Cho em xin link đăng ký xét tuyển của trường?",
        "ground_truth_answer": "Link đăng ký: http://nhaphoc.husc.edu.vn/ và https://husc.edu.vn. Đăng ký trên hệ thống Bộ GDĐT: thisinh.thitotnghiepthpt.edu.vn",
        "gt_chunks": ["husc_info"],
        "must_include": ["husc.edu.vn"],
    },
    {
        "id": "q098",
        "category": "multihop",
        "question": "Em muốn tìm hiểu chính sách học bổng của trường?",
        "ground_truth_answer": "Học bổng theo điểm đầu vào: >=25 điểm miễn 100% học phí HK1, 24-<25 điểm miễn 50%. Học bổng theo kết quả học tập mỗi học kỳ. Học bổng từ doanh nghiệp và tổ chức.",
        "gt_chunks": ["hocbong_2026"],
        "must_include": ["học bổng", "100%", "học phí"],
    },
    {
        "id": "q099",
        "category": "simple",
        "question": "Trường có KTX cho sinh viên không?",
        "ground_truth_answer": "Có, HUSC có ký túc xá với 2 cơ sở (CS1 và CS2). Đăng ký online tại http://sv-dhh.kytucxa.vn/",
        "gt_chunks": ["husc_info"],
        "must_include": ["ký túc xá"],
    },
    {
        "id": "q100",
        "category": "simple",
        "question": "Trường có group hay zalo tư vấn tuyển sinh, hướng dẫn đăng ký xét tuyển không ạ?",
        "ground_truth_answer": "Không có thông tin trong tài liệu tuyển sinh về group/Zalo tư vấn. Có thể liên hệ số điện thoại 094 4455136 hoặc website husc.edu.vn.",
        "gt_chunks": ["husc_info"],
        "must_include": ["không"],
    },
    {
        "id": "q101",
        "category": "temporal",
        "question": "Khi nào thì trường mình mở hệ thống đki xét tuyển chính thức ạ?",
        "ground_truth_answer": "Thí sinh đăng ký xét tuyển đợt 1 từ 2/7 đến 17h00 ngày 14/7/2026 trên hệ thống Bộ GDĐT. Xác nhận nhập học từ 22/08 đến 17h00 ngày 30/08/2026.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["đăng ký", "2026"],
    },
    {
        "id": "q102",
        "category": "temporal",
        "question": "Hạn cuối xét tuyển là khi nào?",
        "ground_truth_answer": "Hạn cuối xác nhận nhập học trực tuyến: trước 17h00 ngày 21/8/2026 (hoặc 30/08/2026 tùy đợt).",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["17h00", "2026"],
    },
    {
        "id": "q103",
        "category": "multihop",
        "question": "Trường có mở cổng xét tuyển riêng trên hệ thống của trường không?",
        "ground_truth_answer": "Không, thí sinh phải đăng ký trên hệ thống của Bộ GDĐT. HUSC không có cổng xét tuyển riêng. Sau khi trúng tuyển, xác nhận nhập học trên hệ thống Bộ.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["Bộ GDĐT", "không"],
    },
    {
        "id": "q104",
        "category": "multihop",
        "question": "Dạ em chào thầy cô, thầy cô cho em hỏi em thuộc diện khuyết tật nặng, vậy em cần nộp hồ sơ như thế nào ạ?",
        "ground_truth_answer": "Hồ sơ xét tuyển thẳng cho người khuyết tật nặng gồm: Phiếu đăng ký xét tuyển thẳng (Phụ lục 2), bản photocopy Giấy xác nhận người khuyết tật nặng, bản photocopy học bạ THPT.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["khuyết tật", "Phụ lục 2"],
    },
    {
        "id": "q105",
        "category": "multihop",
        "question": "Học ngành lịch sử ở trường mình sau có đi dạy được không?",
        "ground_truth_answer": "Ngành Lịch sử (7229010) đào tạo Cử nhân 4 năm. Với bằng Cử nhân Lịch sử, có cơ hội đi dạy môn Lịch sử ở các trường phổ thông hoặc trung tâm giáo dục.",
        "gt_chunks": ["tuyensinh_ls_2026"],
        "must_include": ["Lịch sử", "dạy"],
    },
    {
        "id": "q106",
        "category": "multihop",
        "question": "Ngành văn học trường mình ra trường có dễ kiếm việc làm không ạ? Nếu học thêm nghiệp vụ sư phạm thì có đi dạy được không?",
        "ground_truth_answer": "Ngành Văn học (7229030) đào tạo Cử nhân 4 năm. Có thể làm việc trong lĩnh vực truyền thông, xuất bản, giáo dục. Nếu học thêm nghiệp vụ sư phạm thì có thể dạy học.",
        "gt_chunks": ["tuyensinh_vanhoc_2026"],
        "must_include": ["Văn học", "dạy"],
    },
    {
        "id": "q107",
        "category": "simple",
        "question": "Trường mình có yêu cầu tiếng anh đầu vào không ạ?",
        "ground_truth_answer": "Không có yêu cầu tiếng Anh đầu vào bắt buộc. Tuy nhiên, thí sinh có chứng chỉ IELTS, TOEFL iBT, VSTEP còn hạn đến 12/08/2026 có thể quy đổi thành điểm thi THPT môn Tiếng Anh.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["chứng chỉ", "quy đổi"],
    },
    {
        "id": "q108",
        "category": "multihop",
        "question": "Cách quy đổi điểm chứng chỉ tiếng anh như thế nào?",
        "ground_truth_answer": "Chứng chỉ IELTS/TOEFL iBT/VSTEP còn hạn đến 12/08/2026 được quy đổi thành điểm THPT môn Tiếng Anh. Cao nhất 10.0 điểm với IELTS >=7.0. Có ít nhất 5 mức điểm khác nhau.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["IELTS", "quy đổi", "10.0"],
    },
    {
        "id": "q109",
        "category": "multihop",
        "question": "E là học sinh trường chuyên có được cộng điểm ưu tiên gì k?",
        "ground_truth_answer": "Học sinh trường chuyên có thể được cộng điểm thưởng nếu đạt giải HSG cấp tỉnh: Nhất +3.0, Nhì +2.5, Ba +2.0. Ngoài ra vẫn được cộng ưu tiên khu vực: KV1 +0.75, KV2-NT +0.5, KV2 +0.25.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["giải", "3.0", "ưu tiên khu vực"],
    },
    {
        "id": "q110",
        "category": "temporal",
        "question": "Trường ơi, dự kiến năm nay điểm có tăng không ạ?",
        "ground_truth_answer": "Không có dự báo chính thức về tăng/giảm điểm chuẩn. Điểm chuẩn phụ thuộc vào chỉ tiêu và chất lượng hồ sơ đăng ký từng năm.",
        "gt_chunks": ["tuyensinh_overview_2026"],
        "must_include": ["điểm chuẩn"],
    },
    {
        "id": "q111",
        "category": "multihop",
        "question": "Em muốn hỏi ngành truyền thông số 1 năm có mấy tín chỉ và học phí bao nhiêu ạ?",
        "ground_truth_answer": "Truyền thông đa phương tiện thuộc khối ngành VII, học phí ~525000 VNĐ/tín chỉ. Trung bình 15-18 tín chỉ/học kỳ, khoảng 30 tín chỉ/năm.",
        "gt_chunks": ["tuyensinh_truyenthong_2026"],
        "must_include": ["525000", "tín chỉ"],
    },
    {
        "id": "q112",
        "category": "multihop",
        "question": "Học bổng xét chung cho tất cả các phương thức hay chỉ xét mỗi điểm thi THPT vậy ad?",
        "ground_truth_answer": "Học bổng không phân biệt phương thức xét tuyển. Điểm thưởng xét cho mọi phương thức dựa trên điểm đầu vào hoặc kết quả học tập.",
        "gt_chunks": ["hocbong_2026"],
        "must_include": ["học bổng", "không phân biệt"],
    },
    {
        "id": "q113",
        "category": "multihop",
        "question": "Trường có bắt buộc sắp xếp thứ tự nguyện vọng không ạ?",
        "ground_truth_answer": "Có, thí sinh cần sắp xếp thứ tự nguyện vọng. Hệ thống xử lý nguyện vọng để xác định nguyện vọng cao nhất mà thí sinh đủ điều kiện trúng tuyển.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["nguyện vọng", "sắp xếp"],
    },
    {
        "id": "q114",
        "category": "temporal",
        "question": "Năm nay trường mình xét học bạ có trừ điểm chênh lệch không hay có hình thức quy đổi điểm k ạ?",
        "ground_truth_answer": "Xét học bạ THPT (mã 200) không nhân hệ số các môn. Điểm xét tuyển là điểm trung bình 4 học kỳ lớp 11 và 12. Được cộng điểm ưu tiên khu vực và điểm thưởng.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["không nhân hệ số", "điểm trung bình"],
    },
    {
        "id": "q115",
        "category": "multihop",
        "question": "Em thí sinh tự do, không đăng ký thi lại THPT năm nay, vậy em có được đăng ký xét tuyển vào trường không?",
        "ground_truth_answer": "Thí sinh tự do không thi lại THPT vẫn có thể đăng ký xét tuyển nếu đã tốt nghiệp THPT. Có thể dùng kết quả thi năm trước hoặc xét học bạ THPT.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["tốt nghiệp", "xét tuyển"],
    },
    {
        "id": "q116",
        "category": "multihop",
        "question": "Em muốn nộp hồ sơ xét tuyển theo điểm đánh giá năng lực HCM ạ.",
        "ground_truth_answer": "HUSC có áp dụng xét tuyển dựa trên điểm đánh giá năng lực. Điểm trúng tuyển tính theo thang 30, cộng đầy đủ điểm ưu tiên và điểm cộng.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["đánh giá năng lực", "xét tuyển"],
    },
    {
        "id": "q117",
        "category": "simple",
        "question": "E muốn xét học bạ nhưng hạnh kiểm Khá có xét được k ạ?",
        "ground_truth_answer": "Có, hạnh kiểm Khá vẫn được xét học bạ. Điều kiện: kết quả trung bình chung học tập mức khá hoặc điểm trung bình các môn lớp 12 từ 6.5 trở lên.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["Khá", "xét"],
    },
    {
        "id": "q118",
        "category": "multihop",
        "question": "Trường có chính sách hỗ trợ sinh viên khó khăn không?",
        "ground_truth_answer": "Có, HUSC có nhiều chính sách: học bổng điểm đầu vào cao (thủ khoa miễn 100% HP HK1), học bổng theo kết quả học tập, hỗ trợ từ quỹ khuyến học, giảm học phí cho đối tượng chính sách.",
        "gt_chunks": ["hocbong_2026", "husc_info"],
        "must_include": ["học bổng", "hỗ trợ"],
    },
    {
        "id": "q119",
        "category": "multihop",
        "question": "Nếu em đậu NV1 mà không muốn học có được xét NV2 không?",
        "ground_truth_answer": "Nếu đã xác nhận nhập học tại NV1 thì không được xét NV2 hoặc các đợt bổ sung, trừ trường hợp được cơ sở đào tạo cho phép. Nếu chưa xác nhận thì vẫn có thể tham gia xử lý nguyện vọng.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["xác nhận nhập học", "không"],
    },
    {
        "id": "q120",
        "category": "multihop",
        "question": "Cho em hỏi là mình đăng ký trên cổng thông tin của Bộ GD có được chọn xét học bạ không ạ? Hay chỉ xét điểm thi THPT quốc gia thôi ạ?",
        "ground_truth_answer": "Có thể chọn xét học bạ THPT (mã 200) khi đăng ký trên cổng Bộ GDĐT, không bắt buộc chỉ xét điểm thi THPT (mã 100). Thí sinh đăng ký tối đa 15 nguyện vọng.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["học bạ", "200"],
    },
    {
        "id": "q121",
        "category": "multihop",
        "question": "Em đặt nguyện vọng của trường mình ở NV3 vậy có khả năng đậu không ạ?",
        "ground_truth_answer": "Có khả năng đậu nếu điểm đủ ngưỡng trúng tuyển. Hệ thống xử lý nguyện vọng lấy nguyện vọng cao nhất mà thí sinh đủ điều kiện.",
        "gt_chunks": ["dieukien_tuyensinh_2026"],
        "must_include": ["nguyện vọng", "trúng tuyển"],
    },
    {
        "id": "q122",
        "category": "temporal",
        "question": "Điểm chuẩn 3 năm gần đây như thế nào ạ, có chênh lệch nhiều không ạ?",
        "ground_truth_answer": "Điểm chuẩn có dao động nhưng nhìn chung không quá lớn. Các ngành khoa học cơ bản thường ổn định hơn, các ngành hot có thể tăng nhẹ.",
        "gt_chunks": ["tuyensinh_overview_2025", "tuyensinh_overview_2026"],
        "must_include": ["điểm chuẩn", "dao động"],
    },
    {
        "id": "q123",
        "category": "simple",
        "question": "Cho em hỏi học phí của trường mình bao nhiêu?",
        "ground_truth_answer": "Học phí theo khối ngành năm 2026: khối VII ~525000 VNĐ/tín chỉ, khối IV ~540000 VNĐ/tín chỉ, khối V ~600000 VNĐ/tín chỉ. Triết học miễn học phí.",
        "gt_chunks": ["tuyensinh_overview_2026"],
        "must_include": ["525000", "540000", "600000"],
    },
]


def main():
    with TEST_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    # Fix q017: strengthen must_include
    for q in questions:
        if q["id"] == "q017":
            q["question"] = "Có bao nhiêu ngành tuyển sinh tại HUSC năm 2026? Liệt kê tất cả các ngành theo thứ tự chỉ tiêu từ cao xuống thấp."
            q["ground_truth_answer"] = "26 ngành đào tạo. Ngành chỉ tiêu cao nhất: CNTT (510), tiếp: KTPM, KHMT, Tin học (120), Toán (100), Hóa (90), Lý (80), Sinh (70), Văn (60)..."
            q["must_include"] = ["26", "Công nghệ thông tin", "510"]
            print("Fixed q017: strengthened must_include")
            break

    # Add new questions
    existing_ids = {q["id"] for q in questions}
    added = 0
    for nq in NEW_QUESTIONS:
        if nq["id"] not in existing_ids:
            questions.append(nq)
            added += 1

    # Sort by id
    def sort_key(q):
        qid = q["id"]
        prefix = qid[0]  # 'q'
        num = int(qid[1:])
        return num

    questions.sort(key=sort_key)

    # Write back
    with TEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Added {added} new questions. Total: {len(questions)}")
    print(f"Categories: simple={sum(1 for q in questions if q['category']=='simple')}, "
          f"multihop={sum(1 for q in questions if q['category']=='multihop')}, "
          f"temporal={sum(1 for q in questions if q['category']=='temporal')}")


if __name__ == "__main__":
    main()
