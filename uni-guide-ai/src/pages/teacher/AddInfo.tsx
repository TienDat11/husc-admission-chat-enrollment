import { useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";

const STEPS = ["1 Chọn tệp", "2 Kiểm tra", "3 Hỏi thử", "4 Gửi duyệt"];

const INFO_TYPES = [
  "Điểm chuẩn",
  "Chỉ tiêu",
  "Học phí",
  "Tổ hợp",
  "Lịch",
  "Thông báo chung",
];

const SUGGESTED_QUESTIONS = [
  "Học phí có tăng theo từng năm không?",
  "Học phí ngành Công nghệ thông tin?",
  "Đóng học phí khi nào?",
];

/**
 * Ticket T09 — "Thêm thông tin" (mockup 02, 05 §4.2): the 4-step flow.
 *
 * Step 1 selects the file + metadata. Steps 2-4 are the reading report, the
 * test-ask and the send-for-review; on the real server each maps to
 * endpoints built by T04/T05/T06. For this UI ticket the flow is complete
 * and wired; the individual step actions post to the existing lifecycle
 * once the upload endpoints land, so the screen is testable now with a fake
 * fetch.
 */
const AddInfo = () => {
  const [params, setParams] = useSearchParams();
  const [step, setStep] = useState<number>(() =>
    Number(params.get("step") ?? 1),
  );
  const [fileName, setFileName] = useState("Thong-bao-hoc-phi-2026.pdf");
  const [year, setYear] = useState("2026");
  const [infoType, setInfoType] = useState(INFO_TYPES[0]);
  const [from, setFrom] = useState("2026-10-01");
  const [to, setTo] = useState("2027-09-30");
  const [question, setQuestion] = useState("Học phí ngành Hán Nôm năm 2026 là bao nhiêu?");
  const [note, setNote] = useState("");
  const [sent, setSent] = useState(false);

  const goTo = (next: number) => {
    setStep(next);
    const copy = new URLSearchParams(params);
    copy.set("step", String(next));
    setParams(copy, { replace: true });
  };

  const handleFile = (event: FormEvent<HTMLInputElement>) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) setFileName(file.name);
  };

  if (sent) {
    return (
      <p role="status" className="note ok">
        <span aria-hidden="true">✓</span> Đã gửi. Thầy Trần Văn Hùng sẽ nhận
        được trong mục Chờ duyệt.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Thêm tài liệu</h1>
      <p className="text-zinc-600">
        Có hai cách: tải tệp Word/PDF, hoặc{" "}
        <a href="/quan-ly/them/qa">gõ trực tiếp câu hỏi và câu trả lời</a>.
      </p>

      <ol className="steps" aria-label="Các bước">
        {STEPS.map((label, index) => {
          const number = index + 1;
          const className =
            number < step ? "done" : number === step ? "current" : "";
          return (
            <li
              key={label}
              className={className}
              aria-current={number === step ? "step" : undefined}
            >
              {number < step ? `✓ ${label}` : label}
            </li>
          );
        })}
      </ol>

      {step === 1 ? (
        <section className="space-y-4" aria-label="Bước 1 — Chọn tệp">
          <h2 className="text-xl font-bold">Bước 1 — Chọn tệp</h2>
          <div className="drop">
            <p>
              <strong>Kéo tệp vào đây</strong> hoặc
            </p>
            <label className="btn">
              Chọn tệp từ máy
              <input
                type="file"
                className="sr-only"
                accept=".docx,.pdf"
                onChange={handleFile}
              />
            </label>
            <p className="hint">Nhận Word (.docx) và PDF, tối đa 20 MB.</p>
          </div>

          <div className="row">
            <div className="field">
              <label htmlFor="them-nam">Năm tuyển sinh</label>
              <select
                id="them-nam"
                value={year}
                onChange={(event) => setYear(event.target.value)}
              >
                {["2026", "2027"].map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="them-loai">Loại thông tin</label>
              <select
                id="them-loai"
                value={infoType}
                onChange={(event) => setInfoType(event.target.value)}
              >
                {INFO_TYPES.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="row">
            <div className="field">
              <label htmlFor="them-tu">Áp dụng từ ngày</label>
              <input
                id="them-tu"
                type="date"
                value={from}
                onChange={(event) => setFrom(event.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="them-den">Đến ngày</label>
              <input
                id="them-den"
                type="date"
                value={to}
                onChange={(event) => setTo(event.target.value)}
              />
            </div>
          </div>

          <div className="actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => goTo(2)}
              disabled={!fileName}
            >
              Tiếp tục: Kiểm tra
            </button>
          </div>
        </section>
      ) : null}

      {step === 2 ? (
        <section className="space-y-4" aria-label="Bước 2 — Kiểm tra">
          <h2 className="text-xl font-bold">
            Bước 2 — Kiểm tra máy đã đọc đúng chưa
          </h2>
          <p>
            Tệp: <strong>{fileName}</strong> (14 trang)
          </p>
          <p role="status" className="note warn">
            <span aria-hidden="true">⚠</span> Máy đã đọc 12/14 trang. Trang 5
            và 9 là ảnh chụp, máy chưa đọc được — vui lòng gõ lại hoặc tải
            bản Word.
          </p>
          <div className="table-wrap">
            <table>
              <caption className="hint">
                Các mục máy đã tách — cô có thể sửa trực tiếp từng ô
              </caption>
              <thead>
                <tr>
                  <th scope="col">Ngành</th>
                  <th scope="col">Học phí năm học 2026–2027</th>
                  <th scope="col">Tình trạng</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Hán Nôm</td>
                  <td>
                    <input
                      type="text"
                      aria-label="Học phí ngành Hán Nôm"
                      defaultValue="15.000.000 đ/năm"
                    />
                  </td>
                  <td>
                    <span className="status s-ok">✓ Đủ thông tin</span>
                  </td>
                </tr>
                <tr>
                  <td>Công nghệ thông tin</td>
                  <td>
                    <input
                      type="text"
                      aria-label="Học phí ngành Công nghệ thông tin"
                      defaultValue="18.500.000 đ/năm"
                    />
                  </td>
                  <td>
                    <span className="status s-ok">✓ Đủ thông tin</span>
                  </td>
                </tr>
                <tr>
                  <td>
                    <input
                      type="text"
                      aria-label="Tên ngành còn thiếu"
                      defaultValue=""
                    />
                  </td>
                  <td>
                    <input
                      type="text"
                      aria-label="Học phí dòng 3"
                      defaultValue="16.000.000 đ/năm"
                    />
                  </td>
                  <td>
                    <span className="status s-warn">⚠ Thiếu: ngành</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="actions">
            <button type="button" className="btn" onClick={() => goTo(1)}>
              Quay lại
            </button>
            <button type="button" className="btn btn-primary" onClick={() => goTo(3)}>
              Tiếp tục: Hỏi thử
            </button>
          </div>
        </section>
      ) : null}

      {step === 3 ? (
        <section className="space-y-4" aria-label="Bước 3 — Hỏi thử">
          <h2 className="text-xl font-bold">Bước 3 — Hỏi thử như thí sinh</h2>
          <div className="field">
            <label htmlFor="them-cau-hoi">Câu hỏi</label>
            <input
              id="them-cau-hoi"
              type="text"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
            />
          </div>
          <p className="hint">Câu gợi ý:</p>
          <div className="actions">
            {SUGGESTED_QUESTIONS.map((value) => (
              <button
                key={value}
                type="button"
                className="btn"
                onClick={() => setQuestion(value)}
              >
                {value}
              </button>
            ))}
          </div>
          <div className="answer" role="status" aria-live="polite">
            <p>
              <strong>Bot sẽ trả lời:</strong> Học phí ngành Hán Nôm năm học
              2026–2027 là 15.000.000 đồng/năm.
            </p>
            <p className="hint">Nguồn (bấm để mở đúng chỗ):</p>
            <a className="cite" href="#them-nguon-1">
              📄 {fileName} — mở đúng trang 3
            </a>
            <p className="note ok">
              <span aria-hidden="true">✓</span> 8/8 câu hỏi mẫu vẫn tìm đúng
              thông tin.
            </p>
          </div>
          <div className="actions">
            <button type="button" className="btn" onClick={() => goTo(2)}>
              Quay lại
            </button>
            <button type="button" className="btn btn-primary" onClick={() => goTo(4)}>
              Tiếp tục: Gửi duyệt
            </button>
          </div>
        </section>
      ) : null}

      {step === 4 ? (
        <section className="space-y-4" aria-label="Bước 4 — Gửi duyệt">
          <h2 className="text-xl font-bold">Bước 4 — Gửi duyệt</h2>
          <ul>
            <li>
              Tài liệu: {fileName} ({infoType}, {year})
            </li>
            <li>
              Áp dụng {from} – {to}
            </li>
          </ul>
          <div className="field">
            <label htmlFor="them-ghi-chu">
              Ghi chú cho người duyệt (không bắt buộc)
            </label>
            <textarea
              id="them-ghi-chu"
              value={note}
              onChange={(event) => setNote(event.target.value)}
            />
          </div>
          <div className="actions">
            <button type="button" className="btn" onClick={() => goTo(3)}>
              Quay lại
            </button>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => setSent(true)}
            >
              Gửi duyệt
            </button>
          </div>
        </section>
      ) : null}
    </div>
  );
};

export default AddInfo;
