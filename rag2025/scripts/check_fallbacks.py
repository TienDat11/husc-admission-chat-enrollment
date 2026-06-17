import re
_AGGREGATION_RULES = [
    {'patterns': [r'bao\s+nhiêu\s+ngành', r'tất\s+cả\s+ngành', r'danh\s+sách\s+ngành', r'liệt\s+kê.*ngành', r'toàn\s+bộ\s+ngành', r'có\s+mấy\s+ngành', r'các\s+ngành', r'ngành\s+đào\s+tạo', r'ngành\s+học', r'chương\s+trình\s+đào\s+tạo', r'tìm\s+hiểu.*ngành', r'ngành.*trường', r'ngành\s+gì', r'có\s+ngành', r'những\s+ngành'], 'chunks': ['liet_ke_26_nganh_2026','tuyensinh_overview_2026','bang_xep_hang_chi_tieu_2026']},
    {'patterns': [r'khối\s+nào', r'tuyển.*khối', r'những\s+khối', r'các\s+khối', r'khối\s+xét\s+tuyển', r'khối\s+thi'], 'chunks': ['tuyensinh_overview_2026','tohop_a00_full_2026']},
    {'patterns': [r'cổng.*xét\s+tuyển', r'cổng.*riêng', r'cổng.*trường', r'xét\s+tuyển.*riêng.*trường', r'hệ\s+thống.*trường'], 'chunks': ['qa_link_cong_xet_tuyen_husc']},
    {'patterns': [r'điểm\s+cộng.*như\s+thế\s+nào', r'điểm\s+ưu\s+tiên.*tính', r'điểm\s+thưởng', r'ưu\s+tiên.*điểm', r'cộng\s+điểm.*như\s+thế'], 'chunks': ['chinhsach_2026','qa_xet_hoc_ba_quy_trinh_2026']},
    {'patterns': [r'đăng\s+ký.*nguyện\s+vọng', r'nguyện\s+vọng.*đăng\s+ký', r'1\s+trường.*2\s+ngành', r'xét.*tổ\s+hợp.*khác'], 'chunks': ['qa_NV1_NV2_NV3_quy_che']},
    {'patterns': [r'thí\s+sinh\s+tự\s+do.*chuẩn\s+bị', r'tự\s+do.*cần.*chuẩn\s+bị', r'tự\s+do.*xét\s+tuyển.*bằng', r'tự\s+do.*phương\s+thức'], 'chunks': ['qa_xet_hoc_ba_quy_trinh_2026','chinhsach_2026','phuongthuc_xettuyen_2026']},
    {'patterns': [r'học\s+sinh\s+giỏi.*3\s+năm', r'hsg.*3\s+năm', r'giỏi.*cộng\s+điểm', r'thí\s+sinh\s+tự\s+do', r'tự\s+do.*xét\s+tuyển', r'tự\s+do.*học\s+bạ', r'tự\s+do.*tốt\s+nghiệp'], 'chunks': ['chinhsach_2026','qa_xet_hoc_ba_quy_trinh_2026']},
    {'patterns': [r'zalo', r'group.*tư\s+vấn', r'hotline'], 'chunks': ['qa_lien_he_tu_van_husc']},
    {'patterns': [r'nv\d', r'nguyện\s+vọng\s+\d', r'nv1.*nv2', r'nv2.*nv3', r'đăng\s+ký.*2\s+ngành', r'2\s+ngành.*cùng\s+trường', r'nguyện\s+vọng.*ngành'], 'chunks': ['qa_NV1_NV2_NV3_quy_che']},
    {'patterns': [r'tổ\s+hợp\s+a00', r'ngành.*a00', r'a00.*ngành', r'tổ\s+hợp.*xét', r'xét.*tổ\s+hợp'], 'chunks': ['to_hop_a00_full_2026','tohop_a00']},
    {'patterns': [r'điểm\s+cộng', r'ưu\s+tiên', r'chứng\s+chỉ\s+ngoại\s+ngữ', r'học\s+sinh\s+giỏi', r'giỏi.*điểm'], 'chunks': ['chinhsach_2026']},
]

queries = [
    ('msg008', 'Cho e hỏi là đăng kí nguyện vọng là một trường có thể đăng kí 2 ngành được không'),
    ('msg011', 'E là thí sinh tự do, tốt nghiệp năm 2004 thì có được xét học bạ vào trường mình không'),
    ('msg014', 'Em thi tốt nghiệp k thi theo những tổ hợp xét tuyển của ngành, vậy em có đăng kí xét tuyển dc k'),
    ('msg021', 'Cho em hỏi trường tuyển những khối nào'),
    ('msg030', 'xét học bạ nếu cấp 3 mà 3 năm đc học sinh giỏi là đc cộng thêm 3 điểm phải k ạ'),
    ('msg042', 'Điểm cộng, điểm ưu tiên này tính như thế nào ạ'),
    ('msg046', 'Em thí sinh tự do muốn đăng ký xét tuyển vào trường, cần chuẩn bị những gì'),
    ('msg053', 'Trường có mở cổng xét tuyển riêng trên hệ thống của trường không'),
    ('msg055', 'Trường có group hay zalo tư vấn tuyển sinh, hướng dẫn đăng ký xét tuyển không ạ'),
    ('msg060', 'Điểm ưu tiên và điểm thưởng tính như thế nào vậy ạ'),
]

for qid, q in queries:
    matched = []
    for rule in _AGGREGATION_RULES:
        for pat in rule['patterns']:
            if re.search(pat, q.lower()):
                matched.extend(rule['chunks'])
                break
    seen = list(dict.fromkeys(matched))
    status = 'OK' if seen else 'NO MATCH'
    print(f'{qid} [{status}]: chunks={seen} | {q[:70]}')
