# 📋 UI REQUIREMENTS — CHATBOT TUYỂN SINH HUSC
> Tích hợp vào trang: https://tuyensinh.husc.edu.vn/  
> Stack gợi ý: HTML + Tailwind CSS + Vanilla JS (hoặc React)  
> Mô hình phía sau: RAG API endpoint tự host  
> Mục tiêu: Widget chatbot nổi (floating) tích hợp mượt vào giao diện hiện có của trang tuyển sinh HUSC

---

## 1. TỔNG QUAN

Xây dựng một **Floating Chatbot Widget** nhúng vào trang tuyển sinh của Trường Đại học Khoa học, Đại học Huế (HUSC). Widget hoạt động độc lập (không can thiệp layout hiện có), hỗ trợ thí sinh hỏi đáp tự động về thông tin tuyển sinh thông qua mô hình RAG đã xây dựng sẵn.

---

## 2. THIẾT KẾ THƯƠNG HIỆU (Brand Identity)

| Thuộc tính | Giá trị |
|---|---|
| **Màu chủ đạo** | `#003087` (xanh navy HUSC) |
| **Màu phụ / accent** | `#E5A823` (vàng gold) |
| **Màu nền chat** | `#F5F7FA` |
| **Màu bong bóng user** | `#003087` (chữ trắng) |
| **Màu bong bóng bot** | `#FFFFFF` (chữ `#1A1A1A`, shadow nhẹ) |
| **Font chữ** | `Be Vietnam Pro`, fallback: `Segoe UI`, sans-serif |
| **Bo góc (border-radius)** | `16px` cho chat window, `24px` cho bong bóng tin nhắn |
| **Logo / Avatar bot** | Logo HUSC tròn (lấy từ `https://tuyensinh.husc.edu.vn/assets/img/sitename_white.png` hoặc dùng icon graduation cap) |

---

## 3. CẤU TRÚC WIDGET

Widget gồm **2 thành phần chính**:

### 3.1 — Nút Mở Chat (Floating Trigger Button)

- Vị trí: **Fixed**, góc **dưới phải** màn hình, cách mép `24px`
- Kích thước: `60px × 60px`, hình tròn
- Màu nền: `#003087`, icon chat màu trắng (hoặc icon robot/graduation cap)
- **Badge thông báo**: Hiển thị chấm đỏ nhỏ (pulse animation) khi chưa mở lần đầu → gợi ý có chatbot
- **Tooltip**: Hover hiện text "Hỏi đáp tuyển sinh"
- **Animation**: Scale-up nhẹ khi hover (`transform: scale(1.1)`), shadow nổi
- Khi click → mở Chat Window (slide-up animation)

---

### 3.2 — Cửa Sổ Chat (Chat Window)

- Vị trí: Fixed, góc dưới phải, ngay phía trên nút trigger
- Kích thước: `380px × 560px` (desktop) / `100vw × 85vh` (mobile, bottom sheet)
- Border-radius: `16px` (desktop)
- Box-shadow: `0 8px 32px rgba(0,0,0,0.18)`
- **Không che nội dung trang** khi mở

#### 3.2.1 Header

```
┌─────────────────────────────────────────┐
│  🎓 [Avatar HUSC]  Trợ lý Tuyển sinh   ✕│
│                    HUSC • Đang hoạt động │
└─────────────────────────────────────────┘
```

- Nền header: gradient `#003087 → #004BB5`
- Avatar: hình tròn 36px, logo HUSC nền trắng
- Tên bot: **"Trợ lý Tuyển sinh HUSC"** — chữ trắng, bold
- Status: chấm xanh lá + text "Đang hoạt động" — chữ trắng mờ nhỏ
- Nút **✕ (đóng)**: icon X trắng, hover nền trắng 20% opacity
- Nút **⟳ (xóa hội thoại)**: icon trash/reset nhỏ bên cạnh X (optional)

---

#### 3.2.2 Khu vực tin nhắn (Message Area)

- Scroll dọc, `overflow-y: auto`
- Padding: `16px`
- Background: `#F5F7FA`

**Tin nhắn Bot (trái):**
```
[Avatar]  ┌────────────────────────┐
          │ Nội dung trả lời...    │
          └────────────────────────┘
          10:32 AM
```
- Nền: `#FFFFFF`, border: `1px solid #E8ECF0`
- Border-radius: `4px 16px 16px 16px`
- Max-width: 80%

**Tin nhắn User (phải):**
```
          ┌────────────────────────┐
          │ Câu hỏi của user...    │
          └────────────────────────┘
                              10:32 AM
```
- Nền: `#003087`, chữ trắng
- Border-radius: `16px 4px 16px 16px`
- Max-width: 80%

**Typing Indicator (bot đang trả lời):**
- Ba chấm nhảy animated (●●●) bên trái, nền trắng
- Hiển thị khi đang chờ API response

**Tin nhắn chào (Welcome Message):**
Tự động hiển thị khi mở chat lần đầu:
> 👋 Xin chào! Tôi là **Trợ lý Tuyển sinh HUSC**.  
> Tôi có thể giúp bạn tìm hiểu về:  
> • Các ngành đào tạo  
> • Phương thức xét tuyển  
> • Điểm chuẩn & chỉ tiêu  
> • Học bổng & ưu tiên  
> Bạn muốn hỏi gì?

---

#### 3.2.3 Quick Reply Chips (Gợi ý câu hỏi nhanh)

Hiển thị dưới welcome message (và có thể sau mỗi câu trả lời bot):

```
[ 📚 Ngành đào tạo ]  [ 📋 Phương thức xét tuyển ]
[ 🎯 Điểm chuẩn 2025 ]  [ 🎓 Học bổng ]
```

- Dạng chip/pill button, nền trắng, border `#003087`, chữ `#003087`
- Hover: nền `#003087`, chữ trắng
- Border-radius: `20px`
- Font size: `13px`
- Khi click → gửi nội dung chip như tin nhắn user

---

#### 3.2.4 Input Area (Thanh nhập liệu)

```
┌──────────────────────────────────┬──────┐
│ Nhập câu hỏi của bạn...          │  ➤  │
└──────────────────────────────────┴──────┘
```

- Background: `#FFFFFF`, border-top: `1px solid #E8ECF0`
- Padding: `12px 16px`
- **Input field**: `border-radius: 24px`, border `#D1D9E0`, placeholder text xám nhạt
  - Focus: border `#003087`, box-shadow `0 0 0 3px rgba(0,48,135,0.1)`
  - Max-height: `100px` (auto-expand, multiline)
- **Nút Gửi**: hình tròn `40px`, nền `#003087`, icon mũi tên trắng
  - Disabled (input rỗng): nền `#B0B8C1`
  - Hover: nền `#004BB5`
- Gửi bằng **Enter** (Shift+Enter = xuống dòng)
- Dưới input: text nhỏ xám "Powered by HUSC RAG AI"

---

## 4. CHỨC NĂNG & LOGIC

### 4.1 Gửi & nhận tin nhắn

```
User nhập → Hiện tin nhắn user ngay lập tức
         → Disable input + hiện typing indicator
         → Gọi API RAG (POST /api/chat)
         → Nhận response → Ẩn typing, hiện tin bot
         → Scroll xuống cuối tự động
         → Enable input lại
```

**API Request mẫu:**
```json
POST /api/chat
{
  "message": "Câu hỏi của user",
  "session_id": "uuid-session",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**API Response mẫu:**
```json
{
  "answer": "Nội dung trả lời từ RAG...",
  "sources": ["link1", "link2"],
  "suggested_questions": ["Câu gợi ý 1", "Câu gợi ý 2"]
}
```

> ⚠️ Nếu API trả về `suggested_questions`, hiển thị chúng như Quick Reply Chips sau câu trả lời.

### 4.2 Xử lý lỗi

- Timeout (>15s): Hiện thông báo "⚠️ Xin lỗi, phản hồi mất quá lâu. Vui lòng thử lại."
- Lỗi mạng: "❌ Không thể kết nối. Vui lòng kiểm tra internet."
- Lỗi server: "🔧 Hệ thống đang bảo trì. Vui lòng thử lại sau."
- Tất cả lỗi đều có nút **"Thử lại"**

### 4.3 Lưu lịch sử hội thoại

- Lưu vào `sessionStorage` (xóa khi đóng tab)
- Nút reset (🗑️) trong header xóa lịch sử và quay về welcome message

### 4.4 Trạng thái mở/đóng

- Lưu trạng thái mở/đóng vào `localStorage` (không reset qua tab)
- Nếu đã mở trước đó → giữ nguyên trạng thái khi reload

---

## 5. RESPONSIVE & MOBILE

| Màn hình | Hành vi |
|---|---|
| Desktop (≥ 768px) | Widget cố định 380×560px, góc dưới phải |
| Mobile (< 768px) | Mở full màn hình dạng bottom sheet (slide-up), chiếm 85vh, border-radius chỉ ở trên `16px` |
| Mobile – Keyboard open | Chat area tự co lại, input luôn visible phía trên keyboard |

---

## 6. ANIMATION & TRANSITIONS

| Sự kiện | Animation |
|---|---|
| Mở chat | Slide-up + fade-in (`300ms ease-out`) |
| Đóng chat | Slide-down + fade-out (`250ms ease-in`) |
| Tin nhắn mới xuất hiện | Fade-in + slight scale từ 0.95→1 (`200ms`) |
| Typing indicator | Ba chấm nhảy lần lượt (keyframe, 1s loop) |
| Nút trigger | Pulse animation nhẹ mỗi 3s để gây chú ý |
| Quick reply chips | Fade-in staggered (mỗi chip delay 50ms) |

---

## 7. ACCESSIBILITY

- Tất cả nút có `aria-label`
- Input có `aria-label="Nhập câu hỏi"`
- Tin nhắn mới được announce qua `aria-live="polite"`
- Keyboard navigation hoạt động đầy đủ (Tab, Enter, Escape để đóng)
- Contrast ratio đảm bảo WCAG AA

---

## 8. KỸ THUẬT TÍCH HỢP VÀO TRANG HIỆN CÓ

Widget được đóng gói thành **1 file HTML/JS duy nhất** (hoặc component độc lập), nhúng vào cuối thẻ `<body>` của trang hiện tại bằng:

```html
<!-- HUSC Chatbot Widget -->
<script src="/chatbot/widget.js"></script>
```

Hoặc nhúng trực tiếp bằng 1 thẻ `<div id="husc-chatbot"></div>` + script init.

**Không dùng iframe.** Không ảnh hưởng CSS/JS của trang gốc (scoped styles).

---

## 9. FILE CẦN TẠO

```
chatbot/
├── index.html          ← Demo standalone (để test)
├── widget.js           ← Script nhúng vào trang
├── widget.css          ← Style scoped (nếu tách riêng)
└── README.md
```

---

## 10. ĐIỂM ĐẶC BIỆT CẦN CHÚ Ý

1. **Màu HUSC chuẩn**: Navy `#003087` + Gold `#E5A823` — không dùng màu khác
2. **Tiếng Việt**: Toàn bộ UI bằng tiếng Việt, font hỗ trợ dấu tiếng Việt
3. **Logo HUSC**: Dùng làm avatar cho bot message và header
4. **Hotline fallback**: Nếu bot không trả lời được → hiển thị "Liên hệ hotline: **0385 887 111** (Zalo/ĐT)"
5. **Không popup quá sớm**: Chỉ auto-open nếu user đã visit trang > 30s hoặc scroll xuống > 50%v
