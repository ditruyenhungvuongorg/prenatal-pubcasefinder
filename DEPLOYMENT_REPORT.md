# Báo cáo triển khai và kiểm thử — 19/09/2026

## Địa chỉ
- Web: https://ditruyenhungvuong.github.io/prenatal-pubcasefinder/
- Repo: https://github.com/ditruyenhungvuong/prenatal-pubcasefinder
- Backend Ubuntu: https://adminpc-system-product-name.taila6ff46.ts.net
- Truy cập API cần mã riêng; mã không nằm trong GitHub, localStorage hoặc tài liệu công khai.

## Đã thực hiện
- Hai cách nhập: tra cứu HPO và trích cụm nguyên văn bằng Model 1 v3.8 thật.
- Chọn gợi ý vào cùng danh sách, chống trùng mã, sửa trạng thái Có/Nghi ngờ/Không.
- Bác sĩ xác nhận danh sách trước khi xếp hạng.
- Vô hiệu hóa kết quả và xuất phiếu khi danh sách hoặc văn bản thay đổi.
- Đối chiếu 12.935 hồ sơ; kiểu di truyền đọc trực tiếp aspect I.
- Thẻ khớp phân biệt chính xác/ngữ nghĩa; loại bằng chứng không khớp, âm tính của hồ sơ.
- Mở dấu hiệu chưa ghi nhận; chọn bệnh đưa vào phiếu; xem trước và nút in/lưu PDF.
- Bộ lọc kiểu di truyền; giao diện responsive; giữ ca trong bộ nhớ tab.
- Dịch vụ systemd người dùng, bật linger, HTTPS qua Tailscale Funnel.
- Tái sử dụng worker v3.8 đã đánh giá; không đổi file train hoặc adapter.
- Chỉ gợi ý xét nghiệm từ hồ sơ có mã bệnh khớp chính xác. Đã chặn lỗi ghép nhầm do từ chung "syndrome".
- GitHub không chứa bệnh án, gold benchmark, Excel, trọng số, bảng HPO hoặc mã truy cập.

## Bằng chứng kiểm thử
- 5 kiểm thử tích hợp web PASS (gồm hồi quy không gán xét nghiệm Turner cho OMIM:109400).
- 11 kiểm thử Stage 2 có sẵn PASS.
- API thật: thiếu mã truy cập -> 401; input sai/rỗng/chỉ âm tính -> 422.
- GPU thật: câu tổng hợp "Thai có đầu nhỏ và hàm dưới nhỏ." -> 2 cụm nguyên văn và tọa độ đúng, gợi ý HP:0000252 / HP:0000347. Một lần đo 0,35 giây, không phải benchmark tổng quát.
- Xếp hạng thật cho HP:0009729 -> 20 kết quả; một lần đo khoảng 2,06 giây.
- UI: tìm mã, chọn HPO, duyệt, xếp hạng, xem phiếu có mã ca, đổi trạng thái khóa xuất phiếu.
- UI: trích đoạn văn thật, chọn 2 HPO và giữ HPO đã nhập tay -> 3 mã; xếp hạng lại thành công.
- Mobile 390px: không tràn ngang; quan sát trực quan vùng nhập và danh sách.
- Không thấy lỗi JavaScript trong luồng QA đã thử.
- Trang GitHub Pages tải thành công. API HTTPS trả model_ready=true.

## Phạm vi chưa xác nhận
- Trình duyệt tích hợp Codex chặn trực tiếp miền Tailscale (ERR_BLOCKED_BY_CLIENT).
  Vì vậy kiểm thử tương tác dùng bản giao diện cục bộ, proxy chuyển tiếp đến API HTTPS thật.
  Chưa chứng minh luồng liên miền trên mọi trình duyệt của bác sĩ.
- Phiếu đã kiểm tra nội dung và xem trước; chưa xác minh bản PDF cuối cùng trên máy in/trình duyệt của người dùng.
- Chưa thử tải đồng thời nhiều bác sĩ, mất điện, khởi động lại máy hoặc huấn luyện song song.
- Chưa đánh giá lại F1 hay xác thực lâm sàng. Các phép thử trên là kiểm thử phần mềm.
- Máy Ubuntu phải bật, có mạng và đủ VRAM. Training có thể cần dừng dịch vụ web để giải phóng GPU.
- Kiểu di truyền không có trong nguồn vẫn để trống; liên kết gen không đồng nghĩa bằng chứng gây bệnh.
- Không tự động quảng bá checkpoint mới thành model đang phục vụ.

## Vận hành trên Ubuntu
Thư mục chạy: ~/prenatal_pubcasefinder/releases/web-20260919
Cấu hình riêng: ~/prenatal_pubcasefinder/.deployment/web.env

Kiểm tra:
```bash
systemctl --user status prenatal-web
journalctl --user -u prenatal-web -n 50 --no-pager
tailscale funnel status
```

Trước train nếu cần toàn bộ GPU:
```bash
systemctl --user stop prenatal-web
```
Sau train, phục vụ lại adapter cũ đã xác nhận:
```bash
systemctl --user start prenatal-web
```

Đổi adapter: kiểm tra riêng bản mới trước; ghi lại đường dẫn cũ; sửa MODEL1_ADAPTER trong
web.env; restart prenatal-web; kiểm tra trích đoạn văn và xếp hạng. Nếu lỗi, trả lại
đường dẫn cũ rồi restart. Không sửa đè trọng số đang được dịch vụ sử dụng.


## Cập nhật giao diện và đồng bộ HPO — 20/09/2026

- Bố cục một cột: thông tin ca, nhập dấu hiệu, HPO đã chọn, kết quả, phiếu Lab.
- Font Be Vietnam Pro tự phục vụ (Regular/SemiBold); giấy phép SIL OFL kèm trong docs/fonts.
- Hai cách nhập dùng chung CaseSelection: chống trùng ID, giữ trạng thái bác sĩ chọn,
  ghi nguồn nhập, xử lý mâu thuẫn trước khi duyệt. Tách lại không xóa HPO đã chọn.
- Hủy/bỏ phản hồi tìm kiếm cũ; bỏ phản hồi trích đoạn văn đã sửa; kết quả đối chiếu
  không còn hiệu lực sau thay đổi ca. Đổi máy chủ không chuyển mã truy cập cũ sang địa chỉ mới.
- Kiểm thử: 7/7 Node state tests; 5/5 backend unittest.
- Trình duyệt qua proxy QA cục bộ tới HTTPS Ubuntu thật: tìm HP:0000252, trích
  "Thai nghi ngờ đầu nhỏ.", xử lý CÓ/NGHI NGỜ, giữ quyết định khi trích lại,
  đồng bộ thay đổi trạng thái sang tìm kiếm, xóa HPO, sửa văn bản, đối chiếu 20 bệnh,
  xem trước phiếu Lab đúng trạng thái. Không thấy console error.
- Kiểm tra trực quan desktop và viewport 390x844: 5 phần cùng chiều rộng,
  không tràn ngang, font tiếng Việt hiển thị đúng.
- Giới hạn: đây là kiểm thử phần mềm, không phải thẩm định lâm sàng; chưa kiểm tra
  bản PDF in thực tế. IAB trước đó chặn truy cập trực tiếp tên miền ts.net nên kiểm thử
  luồng API trong UI qua proxy; kiểm tra HTTPS/CORS trực tiếp bằng HTTP client riêng.
