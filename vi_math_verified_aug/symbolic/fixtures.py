"""Hand-written IRs for real seeds (23 Vi-GSM8K train + 5 Vi-MATH algebra train).

Texts are byte-identical copies of data/translation/gsm8k/gsm8k_train_vietnamese.jsonl and
data/translation/MATH/train/algebra_vi.jsonl (tests check this when the files exist). Every fixture passes
validate_program(P, gold). They double as reference outputs for the formalizer prompt and as the test bed
for the operators and rejection rules.

    load_fixture("gsm8k-0") -> Program
    all_fixtures()          -> {seed_id: Program}
"""
from __future__ import annotations

from .ir import Program, ProgramBuilder

__all__ = ["SEEDS", "load_fixture", "all_fixtures", "GSM8K_IDS", "ALGEBRA_IDS"]

SEEDS: dict[str, tuple[str, str]] = {
    "gsm8k-0": ('Natalia đã bán kẹp tóc cho 48 người bạn của mình vào tháng 4, và sau đó cô ấy đã bán số kẹp tóc bằng một nửa số đó vào tháng 5. Natalia đã bán tổng cộng bao nhiêu kẹp tóc vào tháng 4 và tháng 5?', "72"),
    "gsm8k-1": ('Weng kiếm được 12 đô la mỗi giờ khi trông trẻ. Hôm qua, cô ấy chỉ trông trẻ 50 phút. Cô ấy đã kiếm được bao nhiêu tiền?', "10"),
    "gsm8k-2": ('Betty đang tiết kiệm tiền để mua một chiếc ví mới có giá 100 đô la. Betty chỉ có một nửa số tiền cô ấy cần. Bố mẹ cô ấy quyết định cho cô ấy 15 đô la cho mục đích đó, và ông bà cô ấy cho gấp đôi số tiền bố mẹ cô ấy cho. Betty cần thêm bao nhiêu tiền nữa để mua chiếc ví?', "5"),
    "gsm8k-3": ('Julie đang đọc một cuốn sách 120 trang. Hôm qua, cô ấy đã đọc được 12 trang và hôm nay, cô ấy đã đọc gấp đôi số trang hôm qua. Nếu cô ấy muốn đọc một nửa số trang còn lại vào ngày mai, cô ấy nên đọc bao nhiêu trang?', "42"),
    "gsm8k-5": ('Mark có một khu vườn với các loài hoa. Anh ấy đã trồng cây với ba màu khác nhau trong đó. Mười cây màu vàng, và số cây màu tím nhiều hơn 80%. Số hoa màu xanh lá cây chỉ bằng 25% tổng số hoa màu vàng và màu tím. Mark có bao nhiêu bông hoa trong vườn của mình?', "35"),
    "gsm8k-7": ('Ken đã tạo một gói quà để gửi cho em trai mình, người đang học ở trường nội trú. Ken đặt một cái hộp lên cân, sau đó đổ đủ kẹo dẻo vào hộp để trọng lượng lên đến 2 pound. Sau đó, anh ấy thêm đủ bánh brownie để trọng lượng tăng gấp ba. Tiếp theo, anh ấy thêm 2 pound kẹo dẻo nữa. Và cuối cùng, anh ấy thêm đủ kẹo dẻo hình giun để trọng lượng tăng gấp đôi một lần nữa. Trọng lượng cuối cùng của hộp quà, tính bằng pound, là bao nhiêu?', "16"),
    "gsm8k-10": ('Một con quái vật biển sâu trồi lên từ dưới nước cứ sau một trăm năm để ăn thịt một con tàu và thỏa mãn cơn đói của nó. Trong hơn ba trăm năm, nó đã tiêu thụ 847 người. Các con tàu đã được đóng lớn hơn theo thời gian, vì vậy mỗi con tàu mới có số người gấp đôi con tàu trước. Có bao nhiêu người trên con tàu mà con quái vật đã ăn trong một trăm năm đầu tiên?', "121"),
    "gsm8k-13": ('Jasper sẽ phục vụ charcuterie tại bữa tiệc tối của mình. Anh ấy mua 2 pound phô mai cheddar với giá 10 đô la, một pound phô mai kem có giá bằng một nửa giá phô mai cheddar, và một gói thịt nguội có giá gấp đôi giá phô mai cheddar. Anh ấy chi bao nhiêu tiền cho các nguyên liệu?', "35"),
    "gsm8k-20": ('Bella đã mua tem tại bưu điện. Một số con tem có thiết kế bông tuyết, một số có thiết kế xe tải và một số có thiết kế hoa hồng. Bella đã mua 11 con tem bông tuyết. Cô ấy mua nhiều hơn 9 con tem xe tải so với tem bông tuyết, và ít hơn 13 con tem hoa hồng so với tem xe tải. Bella đã mua tổng cộng bao nhiêu con tem?', "38"),
    "gsm8k-25": ('Ralph sẽ tập chơi tennis với một máy bắn bóng tennis để Ralph đánh. Anh ấy nạp 175 quả bóng tennis vào máy lúc đầu. Trong 100 quả bóng đầu tiên, anh ấy đánh trúng được 2/5 số đó. Trong 75 quả bóng tennis tiếp theo, anh ấy đánh trúng được 1/3 số đó. Trong tất cả các quả bóng tennis, Ralph đã không đánh trúng bao nhiêu quả?', "110"),
    "gsm8k-30": ('Ann, Bill, Cate và Dale mỗi người mua một chiếc pizza chảo cá nhân được cắt thành 4 miếng. Nếu Bill và Dale ăn 50% pizza của họ và Ann và Cate ăn 75% pizza, thì còn lại bao nhiêu miếng pizza chưa ăn?', "6"),
    "gsm8k-40": ('Một vé hòa nhạc có giá 40 đô la. Ông Benson đã mua 12 vé và nhận được chiết khấu 5% cho mỗi vé mua vượt quá 10 vé. Ông Benson đã trả tổng cộng bao nhiêu?', "476"),
    "gsm8k-50": ('Gerald chi 100 đô la mỗi tháng cho các vật tư bóng chày. Mùa giải của anh ấy kéo dài 4 tháng. Anh ấy muốn sử dụng những tháng không chơi bóng chày để tiết kiệm bằng cách dọn lá, xúc tuyết và cắt cỏ. Anh ấy tính phí 10 đô la cho mỗi công việc. Anh ấy cần trung bình bao nhiêu công việc mỗi tháng để tiết kiệm tiền cho vật tư của mình?', "5"),
    "gsm8k-60": ('Hai cô con gái của Jerry chơi bóng mềm trong các đội khác nhau. Mỗi người có 8 trận đấu trong mùa giải này. Mỗi đội tập luyện 4 giờ cho mỗi trận đấu. Nếu mỗi trận đấu kéo dài 2 giờ, Jerry sẽ dành bao nhiêu giờ ở sân để xem các con gái mình chơi và tập luyện tổng cộng?', "96"),
    "gsm8k-70": ('Tim có ít hơn Martha 30 quả táo, và Harry có số táo bằng một nửa Tim. Nếu Martha có 68 quả táo, Harry có bao nhiêu quả táo?', "19"),
    "gsm8k-80": ('Jennifer mua 40 lon sữa ở cửa hàng trước khi gặp Mark, bạn cùng lớp của cô, người cũng đang mua sữa. Jennifer mua thêm 6 lon cho mỗi 5 lon Mark mua. Nếu Mark mua 50 lon, Jennifer đã mang về nhà bao nhiêu lon sữa từ cửa hàng?', "100"),
    "gsm8k-100": ('Một cửa hàng thủ công mỹ nghệ có một phần ba doanh số bán hàng từ bộ phận vải, một phần tư doanh số bán hàng từ bộ phận trang sức, và phần còn lại từ bộ phận văn phòng phẩm. Hôm nay họ đã bán được 36 món hàng. Có bao nhiêu món hàng được bán ở bộ phận văn phòng phẩm?', "15"),
    "gsm8k-150": ('Angela là một người đưa thư bằng xe đạp ở New York. Cô ấy cần giao số gói hàng gấp 8 lần số bữa ăn. Nếu cô ấy cần giao tổng cộng 27 bữa ăn và gói hàng, hỏi cô ấy giao bao nhiêu bữa ăn?', "3"),
    "gsm8k-200": ('Sansa là một nghệ sĩ nổi tiếng, cô ấy có thể vẽ một bức chân dung và bán nó tùy theo kích thước. Cô ấy bán một bức chân dung 8 inch với giá $5, và một bức chân dung 16 inch với giá gấp đôi bức chân dung 8 inch. Nếu cô ấy bán ba bức chân dung 8 inch và năm bức chân dung 16 inch mỗi ngày, hỏi cô ấy kiếm được bao nhiêu tiền mỗi 3 ngày?', "195"),
    "gsm8k-250": ('Tom sử dụng 10 đĩa tạ, mỗi đĩa nặng 30 pound, trên một máy tập thể dục. Máy tập này sử dụng công nghệ đặc biệt để làm cho các tạ nặng hơn 20% khi hạ xuống. Các tạ nặng bao nhiêu khi được hạ xuống?', "360"),
    "gsm8k-300": ('Chi phí in một trang là $0.10. Jenny muốn in 7 bản của bài luận 25 trang của mình. Cô ấy cũng muốn mua 7 cây bút, mỗi cây giá $1.50. Nếu cô ấy trả tiền cho cửa hàng bằng 2 tờ hai mươi đô la, cô ấy nên nhận lại bao nhiêu tiền thừa?', "12"),
    "gsm8k-400": ('Karen đang chuẩn bị ba lô cho một chuyến đi bộ đường dài. Cô ấy mang 20 pound nước, 10 pound thức ăn, và 20 pound dụng cụ. Trong chuyến đi, cô ấy uống 2 pound nước mỗi giờ và ăn 1/3 trọng lượng thức ăn mỗi giờ so với nước mỗi giờ. Cô ấy mang bao nhiêu trọng lượng sau sáu giờ?', "34"),
    "gsm8k-500": ('Joe chơi ném bóng với Derek và Tammy. Anh ấy bắt bóng 23 lần. Derek bắt ít hơn gấp đôi số lần Joe bắt 4 lần. Tammy bắt bóng nhiều hơn một phần ba số lần Derek bắt 16 lần. Hỏi Tammy bắt bóng bao nhiêu lần?', "30"),
    "algebra-52": ('Tổng của hai số là $45$. Hiệu của chúng là $3$. Số nhỏ hơn trong hai số đó là bao nhiêu?', "21"),
    "algebra-73": ('Giải $x$: $\\dfrac{1}{2} + \\dfrac{1}{x} = \\dfrac{5}{6}$.', "3"),
    "algebra-69": ('Giá trị của $(2x + 5)^2$ khi $x = 3$ là bao nhiêu?', "121"),
    "algebra-105": ('Cho biết $-7$ là một nghiệm của $x^2 + bx -28 = 0$, giá trị của $b$ là bao nhiêu?', "3"),
    "algebra-145": ('Nếu $F(a, b, c, d) = a^b + c \\times d$, giá trị của $x$ là bao nhiêu sao cho $F(2, x, 4, 11) = 300$?', "8"),
}


def _b(seed_id: str) -> ProgramBuilder:
    text, _ = SEEDS[seed_id]
    return ProgramBuilder(seed_id, text, kind="symbolic" if seed_id.startswith("algebra") else "story")


def _gold(seed_id: str) -> str:
    return SEEDS[seed_id][1]


# ------------------------------------------------------------------------------------------ GSM8K
def _g0():
    b = _b("gsm8k-0")
    a = b.given("48", "count", "kẹp tóc", "số kẹp tóc Natalia bán trong tháng 4")
    h = b.given("một nửa", "ratio", "", "tỉ lệ số kẹp tóc tháng 5 so với tháng 4")
    m = b.op("ratio_part", [a, h], "count", "kẹp tóc", "số kẹp tóc Natalia bán trong tháng 5", "bằng một nửa")
    t = b.op("add", [a, m], "count", "kẹp tóc", "tổng số kẹp tóc Natalia bán trong tháng 4 và tháng 5", "tổng cộng")
    return b.build(t, _gold("gsm8k-0"))


def _g1():
    b = _b("gsm8k-1")
    r = b.given("12", "rate", "đô la/giờ", "số tiền Weng kiếm được mỗi giờ")
    m = b.given("50", "measure", "phút", "số phút Weng trông trẻ hôm qua")
    c = b.const(60, "rate", "phút/giờ", "số phút trong một giờ")
    pm = b.op("div", [r, c], "rate", "đô la/phút", "số tiền Weng kiếm được mỗi phút")
    e = b.op("mul", [pm, m], "money", "đô la", "số tiền Weng kiếm được hôm qua", "kiếm được")
    return b.build(e, _gold("gsm8k-1"))


def _g2():
    b = _b("gsm8k-2")
    price = b.given("100", "money", "đô la", "giá chiếc ví mới")
    half = b.given("một nửa", "ratio", "", "phần số tiền Betty đang có so với giá ví")
    par = b.given("15", "money", "đô la", "số tiền bố mẹ cho Betty")
    two = b.given("gấp đôi", "ratio", "", "số lần tiền ông bà cho so với bố mẹ")
    has = b.op("ratio_part", [price, half], "money", "đô la", "số tiền Betty có lúc đầu")
    gp = b.op("mul", [par, two], "money", "đô la", "số tiền ông bà cho Betty", "cho gấp đôi")
    need = b.op("sub", [price, has, par, gp], "money", "đô la", "số tiền Betty còn cần thêm", "cần thêm")
    return b.build(need, _gold("gsm8k-2"))


def _g3():
    b = _b("gsm8k-3")
    tot = b.given("120", "count", "trang", "số trang của cuốn sách")
    y = b.given("12", "count", "trang", "số trang Julie đọc hôm qua")
    two = b.given("gấp đôi", "ratio", "", "số lần số trang hôm nay so với hôm qua")
    half = b.given("một nửa", "ratio", "", "phần số trang còn lại Julie muốn đọc ngày mai")
    td = b.op("mul", [y, two], "count", "trang", "số trang Julie đọc hôm nay", "đọc gấp đôi")
    rd = b.op("add", [y, td], "count", "trang", "số trang Julie đã đọc trong hai ngày")
    rem = b.op("sub", [tot, rd], "count", "trang", "số trang còn lại của cuốn sách", "còn lại")
    tm = b.op("ratio_part", [rem, half], "count", "trang", "số trang Julie nên đọc vào ngày mai", "đọc một nửa")
    return b.build(tm, _gold("gsm8k-3"))


def _g5():
    b = _b("gsm8k-5")
    b.given("ba", "count", "màu", "số màu hoa trong vườn", context="với ba màu", distractor=True)
    y = b.given("Mười", "count", "cây", "số cây hoa màu vàng")
    p80 = b.given("80%", "percent", "%", "phần trăm số cây màu tím nhiều hơn màu vàng")
    p25 = b.given("25%", "percent", "%", "phần trăm số hoa màu xanh so với tổng hoa vàng và tím")
    pu = b.op("pct_inc", [y, p80], "count", "cây", "số cây hoa màu tím", "nhiều hơn 80%")
    yp = b.op("add", [y, pu], "count", "cây", "tổng số hoa màu vàng và màu tím")
    gr = b.op("pct_of", [yp, p25], "count", "cây", "số hoa màu xanh lá cây", "chỉ bằng 25%")
    tot = b.op("add", [yp, gr], "count", "cây", "tổng số hoa trong vườn của Mark")
    return b.build(tot, _gold("gsm8k-5"))


def _g7():
    b = _b("gsm8k-7")
    w0 = b.given("2", "measure", "pound", "trọng lượng hộp sau khi đổ kẹo dẻo lần đầu", occurrence=0)
    x3 = b.given("gấp ba", "ratio", "", "số lần trọng lượng tăng khi thêm bánh brownie")
    w2 = b.given("2", "measure", "pound", "số pound kẹo dẻo thêm lần hai", occurrence=1)
    x2 = b.given("gấp đôi", "ratio", "", "số lần trọng lượng tăng khi thêm kẹo dẻo hình giun")
    a = b.op("mul", [w0, x3], "measure", "pound", "trọng lượng hộp sau khi thêm bánh brownie", "tăng gấp ba")
    c = b.op("add", [a, w2], "measure", "pound", "trọng lượng hộp sau khi thêm kẹo dẻo lần hai", "thêm")
    d = b.op("mul", [c, x2], "measure", "pound", "trọng lượng cuối cùng của hộp quà", "tăng gấp đôi")
    return b.build(d, _gold("gsm8k-7"))


def _g10():
    b = _b("gsm8k-10")
    period = b.given("một trăm", "measure", "năm", "số năm giữa hai lần quái vật trồi lên", occurrence=0)
    span = b.given("ba trăm", "measure", "năm", "tổng số năm quái vật đã ăn tàu")
    total = b.given("847", "count", "người", "tổng số người quái vật đã ăn")
    r = b.given("gấp đôi", "ratio", "", "số lần số người trên tàu mới so với tàu trước")
    one = b.const(1, "ratio", "", "một (đơn vị)")
    s = b.unknown("count", "người", "số người trên con tàu trong một trăm năm đầu tiên")
    n = b.op("div", [span, period], "count", "con tàu", "số con tàu quái vật đã ăn")
    rn = b.op("pow", [r, n], "ratio", "", "hệ số số người của con tàu sau con tàu cuối")
    num = b.op("sub", [rn, one], "ratio", "", "tử số của tổng cấp số nhân")
    den = b.op("sub", [r, one], "ratio", "", "mẫu số của tổng cấp số nhân")
    fac = b.op("div", [num, den], "ratio", "", "tổng số người tính theo số người trên tàu đầu tiên")
    eaten = b.op("mul", [s, fac], "count", "người", "tổng số người trên tất cả các con tàu")
    b.equation(eaten, total)
    return b.build(s, _gold("gsm8k-10"))


def _g13():
    b = _b("gsm8k-13")
    b.given("2", "measure", "pound", "số pound phô mai cheddar", distractor=True)
    ch = b.given("10", "money", "đô la", "giá phô mai cheddar")
    half = b.given("một nửa", "ratio", "", "giá phô mai kem so với phô mai cheddar")
    two = b.given("gấp đôi", "ratio", "", "giá thịt nguội so với phô mai cheddar")
    cr = b.op("ratio_part", [ch, half], "money", "đô la", "giá phô mai kem", "bằng một nửa")
    mt = b.op("mul", [ch, two], "money", "đô la", "giá gói thịt nguội", "gấp đôi")
    t = b.op("add", [ch, cr, mt], "money", "đô la", "tổng số tiền Jasper chi cho nguyên liệu")
    return b.build(t, _gold("gsm8k-13"))


def _g20():
    b = _b("gsm8k-20")
    sn = b.given("11", "count", "con tem", "số tem bông tuyết Bella mua")
    d1 = b.given("9", "count", "con tem", "số tem xe tải nhiều hơn tem bông tuyết")
    d2 = b.given("13", "count", "con tem", "số tem hoa hồng ít hơn tem xe tải")
    tr = b.op("add", [sn, d1], "count", "con tem", "số tem xe tải", "nhiều hơn")
    ro = b.op("sub", [tr, d2], "count", "con tem", "số tem hoa hồng", "ít hơn")
    t = b.op("add", [sn, tr, ro], "count", "con tem", "tổng số tem Bella mua")
    return b.build(t, _gold("gsm8k-20"))


def _g25():
    b = _b("gsm8k-25")
    tot = b.given("175", "count", "quả bóng tennis", "số bóng tennis nạp vào máy lúc đầu", distractor=True)
    n1 = b.given("100", "count", "quả bóng tennis", "số bóng trong đợt đầu tiên")
    f1 = b.given("2/5", "ratio", "", "phần bóng đánh trúng trong đợt đầu")
    n2 = b.given("75", "count", "quả bóng tennis", "số bóng trong đợt tiếp theo")
    f2 = b.given("1/3", "ratio", "", "phần bóng đánh trúng trong đợt tiếp theo")
    one = b.const(1, "ratio", "", "toàn bộ số bóng của một đợt")
    m1f = b.op("sub", [one, f1], "ratio", "", "phần bóng đánh trượt trong đợt đầu")
    m1 = b.op("ratio_part", [n1, m1f], "count", "quả bóng tennis", "số bóng đánh trượt trong đợt đầu")
    m2f = b.op("sub", [one, f2], "ratio", "", "phần bóng đánh trượt trong đợt tiếp theo")
    m2 = b.op("ratio_part", [n2, m2f], "count", "quả bóng tennis", "số bóng đánh trượt trong đợt tiếp theo")
    t = b.op("add", [m1, m2], "count", "quả bóng tennis", "tổng số bóng Ralph không đánh trúng")
    b.constraint(f"q{n1} + q{n2} == q{tot}")   # the two batches are all the balls loaded
    return b.build(t, _gold("gsm8k-25"))


def _g30():
    b = _b("gsm8k-30")
    sl = b.given("4", "count", "miếng", "số miếng mỗi chiếc pizza")
    p50 = b.given("50%", "percent", "%", "phần trăm pizza Bill và Dale mỗi người ăn")
    p75 = b.given("75%", "percent", "%", "phần trăm pizza Ann và Cate mỗi người ăn")
    ppl = b.const(4, "count", "người", "số người mua pizza (Ann, Bill, Cate và Dale)")
    bd = b.const(2, "count", "người", "số người ăn 50% pizza (Bill và Dale)")
    ac = b.const(2, "count", "người", "số người ăn 75% pizza (Ann và Cate)")
    tot = b.op("mul", [ppl, sl], "count", "miếng", "tổng số miếng pizza")
    bds = b.op("mul", [bd, sl], "count", "miếng", "số miếng pizza của Bill và Dale")
    bde = b.op("pct_of", [bds, p50], "count", "miếng", "số miếng Bill và Dale ăn")
    acs = b.op("mul", [ac, sl], "count", "miếng", "số miếng pizza của Ann và Cate")
    ace = b.op("pct_of", [acs, p75], "count", "miếng", "số miếng Ann và Cate ăn")
    eat = b.op("add", [bde, ace], "count", "miếng", "tổng số miếng pizza đã ăn")
    left = b.op("sub", [tot, eat], "count", "miếng", "số miếng pizza còn lại chưa ăn", "còn lại")
    return b.build(left, _gold("gsm8k-30"))


def _g40():
    b = _b("gsm8k-40")
    pr = b.given("40", "money", "đô la", "giá một vé hòa nhạc")
    n = b.given("12", "count", "vé", "số vé ông Benson mua")
    d = b.given("5%", "percent", "%", "chiết khấu cho mỗi vé vượt quá 10 vé")
    th = b.given("10", "count", "vé", "số vé không được chiết khấu")
    ex = b.op("sub", [n, th], "count", "vé", "số vé được chiết khấu", "vượt quá")
    dp = b.op("pct_dec", [pr, d], "money", "đô la", "giá một vé sau chiết khấu", "chiết khấu")
    c1 = b.op("mul", [dp, ex], "money", "đô la", "tiền các vé được chiết khấu")
    c0 = b.op("mul", [pr, th], "money", "đô la", "tiền các vé giá gốc")
    t = b.op("add", [c0, c1], "money", "đô la", "tổng số tiền ông Benson trả")
    return b.build(t, _gold("gsm8k-40"))


def _g50():
    b = _b("gsm8k-50")
    sp = b.given("100", "rate", "đô la/tháng", "số tiền Gerald chi mỗi tháng cho vật tư")
    m = b.given("4", "measure", "tháng", "số tháng của mùa giải")
    fee = b.given("10", "rate", "đô la/công việc", "số tiền Gerald tính cho mỗi công việc")
    yr = b.const(12, "measure", "tháng", "số tháng trong một năm")
    save = b.op("mul", [sp, m], "money", "đô la", "số tiền Gerald cần tiết kiệm")
    free = b.op("sub", [yr, m], "measure", "tháng", "số tháng không chơi bóng chày")
    pm = b.op("div", [save, free], "rate", "đô la/tháng", "số tiền cần kiếm mỗi tháng")
    jobs = b.op("div", [pm, fee], "rate", "công việc/tháng", "số công việc trung bình mỗi tháng")
    return b.build(jobs, _gold("gsm8k-50"))


def _g60():
    b = _b("gsm8k-60")
    d = b.given("Hai", "count", "con gái", "số con gái của Jerry")
    g = b.given("8", "count", "trận đấu", "số trận đấu của mỗi con gái")
    pr = b.given("4", "rate", "giờ/trận đấu", "số giờ tập luyện cho mỗi trận đấu")
    gl = b.given("2", "rate", "giờ/trận đấu", "số giờ mỗi trận đấu kéo dài")
    play = b.op("mul", [g, gl], "measure", "giờ", "số giờ xem một con gái thi đấu")
    prac = b.op("mul", [g, pr], "measure", "giờ", "số giờ xem một con gái tập luyện")
    one = b.op("add", [play, prac], "measure", "giờ", "số giờ dành cho một con gái")
    t = b.op("mul", [one, d], "measure", "giờ", "tổng số giờ Jerry ở sân")
    return b.build(t, _gold("gsm8k-60"))


def _g70():
    b = _b("gsm8k-70")
    diff = b.given("30", "count", "quả táo", "số táo Tim có ít hơn Martha")
    half = b.given("một nửa", "ratio", "", "phần số táo của Harry so với Tim")
    ma = b.given("68", "count", "quả táo", "số táo của Martha")
    tim = b.op("sub", [ma, diff], "count", "quả táo", "số táo của Tim", "ít hơn")
    ha = b.op("ratio_part", [tim, half], "count", "quả táo", "số táo của Harry", "bằng một nửa")
    return b.build(ha, _gold("gsm8k-70"))


def _g80():
    b = _b("gsm8k-80")
    j0 = b.given("40", "count", "lon sữa", "số lon sữa Jennifer mua lúc đầu")
    k = b.given("6", "count", "lon sữa", "số lon Jennifer mua thêm mỗi lần")
    per = b.given("5", "count", "lon sữa", "số lon Mark mua cho mỗi lần Jennifer mua thêm")
    mk = b.given("50", "count", "lon sữa", "số lon sữa Mark mua")
    times = b.op("div", [mk, per], "count", "lần", "số lần Jennifer mua thêm")
    extra = b.op("mul", [times, k], "count", "lon sữa", "số lon Jennifer mua thêm", "mua thêm")
    t = b.op("add", [j0, extra], "count", "lon sữa", "số lon sữa Jennifer mang về nhà")
    return b.build(t, _gold("gsm8k-80"))


def _g100():
    b = _b("gsm8k-100")
    f3 = b.given("một phần ba", "ratio", "", "phần doanh số của bộ phận vải")
    f4 = b.given("một phần tư", "ratio", "", "phần doanh số của bộ phận trang sức")
    n = b.given("36", "count", "món hàng", "số món hàng bán được hôm nay")
    fab = b.op("ratio_part", [n, f3], "count", "món hàng", "số món hàng bán ở bộ phận vải")
    jew = b.op("ratio_part", [n, f4], "count", "món hàng", "số món hàng bán ở bộ phận trang sức")
    st = b.op("sub", [n, fab, jew], "count", "món hàng", "số món hàng bán ở bộ phận văn phòng phẩm", "phần còn lại")
    return b.build(st, _gold("gsm8k-100"))


def _g150():
    b = _b("gsm8k-150")
    k = b.given("8", "ratio", "", "số lần số gói hàng so với số bữa ăn")
    tot = b.given("27", "count", "đơn hàng", "tổng số bữa ăn và gói hàng cần giao")
    m = b.unknown("count", "bữa ăn", "số bữa ăn Angela giao")
    p = b.unknown("count", "gói hàng", "số gói hàng Angela giao")
    km = b.op("mul", [m, k], "count", "gói hàng", "số gói hàng tính theo số bữa ăn", "gấp 8 lần")
    s = b.op("add", [p, m], "count", "đơn hàng", "tổng số bữa ăn và gói hàng")
    b.equation(p, km)
    b.equation(s, tot)
    return b.build(m, _gold("gsm8k-150"))


def _g200():
    b = _b("gsm8k-200")
    pr = b.given("5", "money", "đô la", "giá một bức chân dung 8 inch")
    two = b.given("gấp đôi", "ratio", "", "giá bức 16 inch so với bức 8 inch")
    ns = b.given("ba", "count", "bức chân dung", "số bức chân dung 8 inch bán mỗi ngày")
    nb = b.given("năm", "count", "bức chân dung", "số bức chân dung 16 inch bán mỗi ngày")
    days = b.given("3", "measure", "ngày", "số ngày")
    bp = b.op("mul", [pr, two], "money", "đô la", "giá một bức chân dung 16 inch", "gấp đôi")
    rs = b.op("mul", [pr, ns], "money", "đô la", "tiền bán bức 8 inch mỗi ngày")
    rb = b.op("mul", [bp, nb], "money", "đô la", "tiền bán bức 16 inch mỗi ngày")
    day = b.op("add", [rs, rb], "money", "đô la", "tiền Sansa kiếm được mỗi ngày")
    t = b.op("mul", [day, days], "money", "đô la", "tiền Sansa kiếm được sau 3 ngày")
    return b.build(t, _gold("gsm8k-200"))


def _g250():
    b = _b("gsm8k-250")
    n = b.given("10", "count", "đĩa tạ", "số đĩa tạ Tom dùng")
    w = b.given("30", "measure", "pound", "trọng lượng mỗi đĩa tạ")
    p = b.given("20%", "percent", "%", "phần trăm tạ nặng hơn khi hạ xuống")
    tw = b.op("mul", [n, w], "measure", "pound", "tổng trọng lượng các đĩa tạ")
    low = b.op("pct_inc", [tw, p], "measure", "pound", "trọng lượng tạ khi hạ xuống", "nặng hơn 20%")
    return b.build(low, _gold("gsm8k-250"))


def _g300():
    b = _b("gsm8k-300")
    pp = b.given("0.10", "rate", "đô la/trang", "chi phí in một trang")
    cp = b.given("7", "count", "bản", "số bản Jenny muốn in", occurrence=0)
    pg = b.given("25", "count", "trang", "số trang của bài luận")
    npen = b.given("7", "count", "cây bút", "số cây bút Jenny muốn mua", occurrence=1)
    pen = b.given("1.50", "money", "đô la", "giá một cây bút")
    nb = b.given("2", "count", "tờ", "số tờ tiền Jenny trả")
    bill = b.given("hai mươi", "money", "đô la", "mệnh giá mỗi tờ tiền")
    pages = b.op("mul", [cp, pg], "count", "trang", "tổng số trang cần in")
    pc = b.op("mul", [pp, pages], "money", "đô la", "chi phí in")
    pc2 = b.op("mul", [npen, pen], "money", "đô la", "tiền mua bút")
    sp = b.op("add", [pc, pc2], "money", "đô la", "tổng số tiền Jenny chi")
    paid = b.op("mul", [nb, bill], "money", "đô la", "số tiền Jenny đưa cho cửa hàng")
    ch = b.op("sub", [paid, sp], "money", "đô la", "số tiền thừa Jenny nhận lại", "nhận lại")
    return b.build(ch, _gold("gsm8k-300"))


def _g400():
    b = _b("gsm8k-400")
    wa = b.given("20", "measure", "pound", "trọng lượng nước Karen mang", occurrence=0)
    fo = b.given("10", "measure", "pound", "trọng lượng thức ăn Karen mang")
    ge = b.given("20", "measure", "pound", "trọng lượng dụng cụ Karen mang", occurrence=1)
    wr = b.given("2", "rate", "pound/giờ", "lượng nước Karen uống mỗi giờ")
    fr = b.given("1/3", "ratio", "", "tỉ lệ lượng thức ăn so với lượng nước mỗi giờ")
    h = b.given("sáu", "measure", "giờ", "số giờ đi bộ")
    fpr = b.op("ratio_part", [wr, fr], "rate", "pound/giờ", "lượng thức ăn Karen ăn mỗi giờ")
    lr = b.op("add", [wr, fpr], "rate", "pound/giờ", "trọng lượng ba lô giảm mỗi giờ")
    loss = b.op("mul", [lr, h], "measure", "pound", "trọng lượng ba lô giảm sau sáu giờ")
    tot = b.op("add", [wa, fo, ge], "measure", "pound", "trọng lượng ba lô lúc đầu")
    rem = b.op("sub", [tot, loss], "measure", "pound", "trọng lượng Karen mang sau sáu giờ")
    b.constraint(f"q{wr} * q{h} <= q{wa}")     # cannot drink more water than she carries
    b.constraint(f"q{wr} * q{fr} * q{h} <= q{fo}")  # nor eat more food than she carries
    return b.build(rem, _gold("gsm8k-400"))


def _g500():
    b = _b("gsm8k-500")
    j = b.given("23", "count", "lần", "số lần Joe bắt bóng")
    two = b.given("gấp đôi", "ratio", "", "số lần so với Joe")
    less = b.given("4", "count", "lần", "số lần Derek bắt ít hơn gấp đôi của Joe")
    th = b.given("một phần ba", "ratio", "", "phần số lần của Derek")
    more = b.given("16", "count", "lần", "số lần Tammy bắt nhiều hơn một phần ba của Derek")
    dj = b.op("mul", [j, two], "count", "lần", "gấp đôi số lần Joe bắt", "gấp đôi")
    de = b.op("sub", [dj, less], "count", "lần", "số lần Derek bắt bóng", "ít hơn")
    t3 = b.op("ratio_part", [de, th], "count", "lần", "một phần ba số lần Derek bắt")
    ta = b.op("add", [t3, more], "count", "lần", "số lần Tammy bắt bóng", "nhiều hơn")
    return b.build(ta, _gold("gsm8k-500"))


# ------------------------------------------------------------------------------------ MATH algebra
def _a52():
    b = _b("algebra-52")
    s = b.given("45", "abstract", "", "tổng của hai số")
    d = b.given("3", "abstract", "", "hiệu của hai số")
    big = b.unknown("abstract", "", "số lớn hơn")
    small = b.unknown("abstract", "", "số nhỏ hơn")
    ss = b.op("add", [big, small], "abstract", "", "tổng hai số")
    dd = b.op("sub", [big, small], "abstract", "", "hiệu hai số")
    b.equation(ss, s)
    b.equation(dd, d)
    return b.build(small, _gold("algebra-52"))


def _a73():
    b = _b("algebra-73")
    half = b.given("\\dfrac{1}{2}", "abstract", "", "số hạng thứ nhất")
    one = b.given("1", "abstract", "", "tử số của số hạng thứ hai")
    rhs = b.given("\\dfrac{5}{6}", "abstract", "", "vế phải")
    x = b.unknown("abstract", "", "x")
    t2 = b.op("div", [one, x], "abstract", "", "số hạng thứ hai")
    lhs = b.op("add", [half, t2], "abstract", "", "vế trái")
    b.equation(lhs, rhs)
    return b.build(x, _gold("algebra-73"))


def _a69():
    b = _b("algebra-69")
    a = b.given("2", "abstract", "", "hệ số của x", occurrence=0)
    c = b.given("5", "abstract", "", "hằng số trong ngoặc")
    e = b.given("2", "abstract", "", "số mũ", occurrence=1)
    x = b.given("3", "abstract", "", "giá trị của x")
    ax = b.op("mul", [a, x], "abstract", "", "2x")
    inner = b.op("add", [ax, c], "abstract", "", "2x + 5")
    v = b.op("pow", [inner, e], "abstract", "", "giá trị của biểu thức")
    return b.build(v, _gold("algebra-69"))


def _a105():
    b = _b("algebra-105")
    r = b.given("-7", "abstract", "", "nghiệm đã biết")
    e = b.given("2", "abstract", "", "số mũ của x")
    c = b.given("28", "abstract", "", "hằng số")
    z = b.given("0", "abstract", "", "vế phải")
    bb = b.unknown("abstract", "", "hệ số b")
    sq = b.op("pow", [r, e], "abstract", "", "x^2 tại nghiệm")
    bx = b.op("mul", [bb, r], "abstract", "", "bx tại nghiệm")
    lhs = b.op("sub", [b.op("add", [sq, bx], "abstract", "", "x^2 + bx"), c], "abstract", "", "vế trái")
    b.equation(lhs, z)
    return b.build(bb, _gold("algebra-105"))


def _a145():
    b = _b("algebra-145")
    a = b.given("2", "abstract", "", "a")
    c = b.given("4", "abstract", "", "c")
    d = b.given("11", "abstract", "", "d")
    v = b.given("300", "abstract", "", "giá trị của F")
    x = b.unknown("abstract", "", "x")
    p = b.op("pow", [a, x], "abstract", "", "a^b")
    cd = b.op("mul", [c, d], "abstract", "", "c × d")
    f = b.op("add", [p, cd], "abstract", "", "F(2, x, 4, 11)")
    b.equation(f, v)
    return b.build(x, _gold("algebra-145"))


_BUILDERS = {
    "gsm8k-0": _g0, "gsm8k-1": _g1, "gsm8k-2": _g2, "gsm8k-3": _g3, "gsm8k-5": _g5, "gsm8k-7": _g7,
    "gsm8k-10": _g10, "gsm8k-13": _g13, "gsm8k-20": _g20, "gsm8k-25": _g25, "gsm8k-30": _g30,
    "gsm8k-40": _g40, "gsm8k-50": _g50, "gsm8k-60": _g60, "gsm8k-70": _g70, "gsm8k-80": _g80,
    "gsm8k-100": _g100, "gsm8k-150": _g150, "gsm8k-200": _g200, "gsm8k-250": _g250, "gsm8k-300": _g300,
    "gsm8k-400": _g400, "gsm8k-500": _g500,
    "algebra-52": _a52, "algebra-73": _a73, "algebra-69": _a69, "algebra-105": _a105, "algebra-145": _a145,
}
GSM8K_IDS = [k for k in _BUILDERS if k.startswith("gsm8k")]
ALGEBRA_IDS = [k for k in _BUILDERS if k.startswith("algebra")]


def load_fixture(seed_id: str) -> Program:
    return _BUILDERS[seed_id]()


def all_fixtures() -> dict[str, Program]:
    return {k: f() for k, f in _BUILDERS.items()}
