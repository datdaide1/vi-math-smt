"""Semantic types and the Vietnamese unit / classifier lexicon (T1 §2.1).

QType decides integrality, sign and the resampling grid; Unit carries the Vietnamese noun, its classifier
(kept unchanged by every edit), whether it is divisible, and a plausibility range.

    resolve_unit("cái kẹo")      -> Unit(lemma="kẹo", classifier="cái", divisible=False, ...)
    resolve_unit("đô la/giờ")    -> Unit(lemma="đô la", per="giờ", divisible=True)   (a rate)
    default_qtype(unit)          -> QType.COUNT / MONEY / MEASURE / RATE / AGE / PERCENT ...
    grid_for(qtype, unit, v)     -> Fraction   resampling grid (COUNT 1, VND 500, USD 0.01, PERCENT 5, ...)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

__all__ = ["QType", "Unit", "LexEntry", "LEXICON", "CLASSIFIERS", "resolve_unit", "default_qtype",
           "grid_for", "unit_class", "INTEGER_QTYPES", "NONNEG_QTYPES", "requires_integer", "requires_nonneg"]


class QType(str, Enum):
    COUNT = "count"            # indivisible nouns (kẹo, học sinh, quả, con): int >= 0
    MONEY = "money"            # VND multiple of 500 when resampled; USD <= 2 decimals; >= 0
    MEASURE = "measure"        # kg, lít, m, km, giờ, phút: rational > 0
    RATE = "rate"              # derived unit (đồng/kg, km/giờ, quả/ngày): rational >= 0
    PERCENT = "percent"        # the percent number itself (20 for 20%)
    RATIO = "ratio"            # p/q (một phần ba, ba phần tư), multipliers (gấp đôi)
    AGE = "age"                # int in [0, 120]
    TIME_POINT = "time_point"  # clock/date (never resampled)
    ABSTRACT = "abstract"      # MATH: plain numbers, sign/integrality from the problem's constraints

    @classmethod
    def parse(cls, s) -> "QType":
        if isinstance(s, QType):
            return s
        s = str(s or "abstract").strip().lower()
        alias = {"time": "measure", "duration": "measure", "number": "abstract", "fraction": "ratio",
                 "multiplier": "ratio", "price": "money", "speed": "rate", "integer": "count"}
        return cls(alias.get(s, s))


INTEGER_QTYPES = {QType.COUNT, QType.AGE}
NONNEG_QTYPES = {QType.COUNT, QType.MONEY, QType.MEASURE, QType.RATE, QType.AGE, QType.PERCENT, QType.RATIO}

_F = Fraction


@dataclass(frozen=True)
class Unit:
    lemma: str = ""                                       # "kẹo", "nghìn đồng", "kg" ("" = dimensionless)
    classifier: str | None = None                         # "cái", "quả", "con", "chiếc" (kept on edit)
    divisible: bool = True                                # False for COUNT nouns -> integrality
    plausible: tuple[Fraction, Fraction] | None = None    # range for a single stated/derived value
    per: str | None = None                                # rate denominator: "giờ" in "đô la/giờ"

    def __str__(self) -> str:
        head = f"{self.classifier} {self.lemma}".strip() if self.classifier else self.lemma
        return f"{head}/{self.per}" if self.per else head

    @property
    def is_rate(self) -> bool:
        return self.per is not None

    def to_dict(self) -> dict:
        d = {"lemma": self.lemma, "classifier": self.classifier, "divisible": self.divisible,
             "per": self.per}
        d["plausible"] = None if self.plausible is None else [str(self.plausible[0]), str(self.plausible[1])]
        return d

    @staticmethod
    def from_dict(d) -> "Unit":
        if d is None:
            return Unit()
        if isinstance(d, str):
            return resolve_unit(d)
        pl = d.get("plausible")
        return Unit(lemma=str(d.get("lemma") or ""), classifier=d.get("classifier"),
                    divisible=bool(d.get("divisible", True)),
                    plausible=None if pl is None else (Fraction(str(pl[0])), Fraction(str(pl[1]))),
                    per=d.get("per"))


@dataclass(frozen=True)
class LexEntry:
    qtype: QType
    classifier: str | None
    divisible: bool
    plausible: tuple[Fraction, Fraction] | None
    cls: str  # unit class used by the M3 template library


# Vietnamese classifiers (loại từ). A leading classifier marks an indivisible count noun.
CLASSIFIERS = {"cái", "con", "quả", "trái", "chiếc", "bông", "cây", "cuốn", "quyển", "tờ", "tấm", "viên", "hạt",
               "miếng", "bức", "bộ", "đôi", "cặp", "chai", "lon", "hộp", "túi", "gói", "cốc", "ly", "bát", "đĩa",
               "thùng", "bao", "ổ", "củ", "lá", "món", "ngôi", "căn", "người", "bài", "trận", "lần", "cuộn",
               "khúc", "thanh", "cục", "que", "sợi", "tập", "xấp", "chậu", "luống", "bó", "vỉ", "khay", "suất",
               "phần", "lọ", "hũ", "can", "bình", "vé", "cành", "nải", "buồng"}

_C = QType.COUNT
_BIG = (_F(0), _F(10 ** 7))


def _count(cls_: str | None, hi: int = 10 ** 6, cls: str = "count") -> LexEntry:
    return LexEntry(_C, cls_, False, (_F(0), _F(hi)), cls)


def _measure(hi, cls: str) -> LexEntry:
    return LexEntry(QType.MEASURE, None, True, (_F(0), _F(hi)), cls)


def _money(hi, cls: str) -> LexEntry:
    return LexEntry(QType.MONEY, None, True, (_F(0), _F(hi)), cls)


LEXICON: dict[str, LexEntry] = {
    # ---- people
    "người": _count(None, 10 ** 8, "person"), "bạn": _count(None, 10 ** 5, "person"),
    "học sinh": _count(None, 10 ** 5, "person"), "sinh viên": _count(None, 10 ** 5, "person"),
    "giáo viên": _count(None, 10 ** 4, "person"), "công nhân": _count(None, 10 ** 5, "person"),
    "khách": _count(None, 10 ** 5, "person"), "khách hàng": _count(None, 10 ** 6, "person"),
    "cầu thủ": _count(None, 10 ** 4, "person"), "trẻ em": _count(None, 10 ** 5, "person"),
    "em bé": _count(None, 10 ** 3, "person"), "con gái": _count("cô", 20, "person"),
    "con trai": _count("cậu", 20, "person"), "con": _count(None, 20, "person"),
    "thành viên": _count(None, 10 ** 5, "person"), "nhân viên": _count(None, 10 ** 5, "person"),
    # ---- animals
    "gà": _count("con"), "vịt": _count("con"), "bò": _count("con"), "lợn": _count("con"), "heo": _count("con"),
    "chó": _count("con"), "mèo": _count("con"), "cá": _count("con"), "chim": _count("con"), "ngựa": _count("con"),
    "dê": _count("con"), "cừu": _count("con"), "thỏ": _count("con"), "ong": _count("con"), "kiến": _count("con"),
    "nhện": _count("con"), "rùa": _count("con"), "ếch": _count("con"), "sâu": _count("con"),
    # ---- food
    "kẹo": _count("cái"), "kẹo dẻo": _count("cái"), "bánh": _count("cái"), "bánh quy": _count("cái"),
    "bánh nướng": _count("cái"), "bánh mì": _count("ổ"), "bánh ngọt": _count("cái"), "cupcake": _count("cái"),
    "pizza": _count("cái"), "miếng pizza": _count("miếng"), "táo": _count("quả"), "cam": _count("quả"),
    "chuối": _count("quả"), "xoài": _count("quả"), "dưa hấu": _count("quả"), "dâu": _count("quả"),
    "dâu tây": _count("quả"), "nho": _count("quả"), "chanh": _count("quả"), "lê": _count("quả"),
    "đào": _count("quả"), "dừa": _count("quả"), "trứng": _count("quả"), "khoai tây": _count("củ"),
    "cà rốt": _count("củ"), "bắp": _count("bắp"), "kem": _count("cây"), "sữa chua": _count("hộp"),
    "bánh brownie": _count("cái"), "bánh muffin": _count("cái"), "bánh donut": _count("cái"),
    # ---- objects
    "kẹp tóc": _count("cái"), "bút": _count("cây"), "bút chì": _count("cây"), "bút màu": _count("cây"),
    "sách": _count("cuốn"), "vở": _count("quyển"), "truyện": _count("cuốn"), "trang": _count(None, 10 ** 5),
    "tem": _count("con"), "thẻ": _count("tấm"), "vé": _count("tấm"), "ảnh": _count("tấm"),
    "bức tranh": _count("bức"), "tranh": _count("bức"), "bức chân dung": _count("bức"),
    "hoa": _count("bông"), "bông hoa": _count("bông"), "cây": _count("cây"), "cây hoa": _count("cây"),
    "hoa hồng": _count("bông"), "lá": _count("chiếc"), "xe": _count("chiếc"), "xe đạp": _count("chiếc"),
    "ô tô": _count("chiếc"), "xe buýt": _count("chiếc"), "thuyền": _count("chiếc"), "tàu": _count("con"),
    "con tàu": _count(None), "áo": _count("chiếc"), "quần": _count("chiếc"), "giày": _count("đôi"),
    "tất": _count("đôi"), "mũ": _count("chiếc"), "đồ chơi": _count("món"), "búp bê": _count("con"),
    "bi": _count("viên"), "viên bi": _count("viên"), "bóng": _count("quả"), "quả bóng": _count("quả"),
    "bóng tennis": _count("quả"), "bóng bay": _count("quả"), "hộp": _count("cái"), "túi": _count("cái"),
    "chai": _count(None), "lon": _count(None), "lon sữa": _count(None), "cốc": _count("cái"), "ly": _count("cái"),
    "đĩa": _count("cái"), "bát": _count("cái"), "gói": _count(None), "thùng": _count(None), "bao": _count(None),
    "ghế": _count("cái"), "bàn": _count("cái"), "phòng": _count("căn"), "nhà": _count("ngôi"),
    "cửa sổ": _count("cái"), "món hàng": _count("món"), "sản phẩm": _count(None), "gạch": _count("viên"),
    "đĩa tạ": _count("chiếc"), "chậu": _count("cái"), "chậu cây": _count("cái"), "khối": _count(None),
    "mảnh ghép": _count("miếng"), "miếng": _count(None), "lát": _count(None), "cuộn": _count(None),
    "tờ": _count(None), "tờ tiền": _count(None), "đồng xu": _count(None), "viên kẹo": _count("viên"),
    "bài hát": _count(None), "video": _count(None), "email": _count(None), "tin nhắn": _count(None),
    "câu hỏi": _count(None), "bài tập": _count(None), "bài": _count(None), "từ": _count(None, 10 ** 7),
    "điểm": _count(None), "bàn thắng": _count(None), "trận đấu": _count(None), "trận": _count(None),
    "lần": _count(None), "công việc": _count(None), "đơn hàng": _count(None), "bước": _count(None, 10 ** 7),
    "vòng": _count(None), "tầng": _count(None, 200), "lớp": _count(None, 100), "đội": _count(None, 10 ** 3),
    "nhóm": _count(None, 10 ** 4), "hàng": _count(None, 10 ** 4), "cột": _count(None, 10 ** 4),
    # ---- money
    "đồng": _money(10 ** 11, "vnd"), "nghìn đồng": _money(10 ** 8, "vnd_k"), "triệu đồng": _money(10 ** 5, "vnd_m"),
    "vnd": _money(10 ** 11, "vnd"), "đô la": _money(10 ** 8, "usd"), "đô": _money(10 ** 8, "usd"),
    "usd": _money(10 ** 8, "usd"), "$": _money(10 ** 8, "usd"), "euro": _money(10 ** 8, "usd"),
    "bảng": _money(10 ** 8, "usd"), "xu": LexEntry(QType.MONEY, None, False, (_F(0), _F(10 ** 7)), "cent"),
    "cent": LexEntry(QType.MONEY, None, False, (_F(0), _F(10 ** 7)), "cent"),
    # ---- measures
    "kg": _measure(10 ** 6, "mass"), "ki-lô-gam": _measure(10 ** 6, "mass"), "g": _measure(10 ** 7, "mass"),
    "gam": _measure(10 ** 7, "mass"), "tấn": _measure(10 ** 4, "mass"), "pound": _measure(10 ** 6, "mass"),
    "ounce": _measure(10 ** 6, "mass"), "lạng": _measure(10 ** 5, "mass"),
    "lít": _measure(10 ** 6, "volume"), "ml": _measure(10 ** 7, "volume"), "gallon": _measure(10 ** 6, "volume"),
    "cốc nước": _measure(10 ** 4, "volume"),
    "m": _measure(10 ** 7, "length"), "mét": _measure(10 ** 7, "length"), "cm": _measure(10 ** 7, "length"),
    "mm": _measure(10 ** 8, "length"), "km": _measure(10 ** 5, "length"), "dặm": _measure(10 ** 5, "length"),
    "inch": _measure(10 ** 6, "length"), "feet": _measure(10 ** 6, "length"), "foot": _measure(10 ** 6, "length"),
    "yard": _measure(10 ** 6, "length"), "m2": _measure(10 ** 8, "area"), "mét vuông": _measure(10 ** 8, "area"),
    "feet vuông": _measure(10 ** 8, "area"), "ha": _measure(10 ** 6, "area"),
    "giờ": _measure(10 ** 5, "time"), "tiếng": _measure(10 ** 5, "time"), "phút": _measure(10 ** 6, "time"),
    "giây": _measure(10 ** 8, "time"), "ngày": _measure(10 ** 5, "time"), "tuần": _measure(10 ** 4, "time"),
    "tháng": _measure(10 ** 4, "time"), "năm": _measure(10 ** 3, "time"), "độ": _measure(10 ** 4, "temp"),
    "calo": _measure(10 ** 7, "energy"),
    # ---- other
    "tuổi": LexEntry(QType.AGE, None, False, (_F(0), _F(120)), "age"),
    "%": LexEntry(QType.PERCENT, None, True, (_F(0), _F(1000)), "percent"),
    "phần trăm": LexEntry(QType.PERCENT, None, True, (_F(0), _F(1000)), "percent"),
}


def _norm(s: str) -> str:
    import unicodedata
    return " ".join(unicodedata.normalize("NFC", s or "").strip().lower().split())


def lookup(lemma: str) -> LexEntry | None:
    return LEXICON.get(_norm(lemma))


def resolve_unit(s: str | None, qtype: QType | str | None = None) -> Unit:
    """Vietnamese unit/noun phrase -> Unit. Unknown nouns default by qtype (COUNT -> indivisible)."""
    qt = QType.parse(qtype) if qtype is not None else None
    t = _norm(s or "")
    if not t:
        return Unit(divisible=qt not in INTEGER_QTYPES if qt else True)
    if "/" in t:
        num, per = [x.strip() for x in t.split("/", 1)]
        nu = resolve_unit(num)
        return Unit(lemma=nu.lemma, classifier=nu.classifier, divisible=True, plausible=None, per=per)
    for m in (" mỗi ", " một "):  # "đô la mỗi giờ" -> rate
        if m in f" {t} " and not t.startswith(m.strip()):
            a, b = t.split(m.strip(), 1)
            if a.strip() and b.strip():
                nu = resolve_unit(a.strip())
                return Unit(lemma=nu.lemma, classifier=nu.classifier, divisible=True, per=b.strip())
    e = lookup(t)
    if e is not None:
        div = e.divisible if qt is None or qt == e.qtype else (qt not in INTEGER_QTYPES)
        return Unit(lemma=t, classifier=e.classifier, divisible=div, plausible=e.plausible)
    toks = t.split()
    if len(toks) >= 2 and toks[0] in CLASSIFIERS:
        rest = " ".join(toks[1:])
        e = lookup(rest)
        pl = e.plausible if e else (_F(0), _F(10 ** 7))
        div = e.divisible if e else False
        if qt is not None and qt not in INTEGER_QTYPES and qt != QType.COUNT:
            div = True
        return Unit(lemma=rest, classifier=toks[0], divisible=div, plausible=pl)
    # longest known suffix/prefix match: "quả táo đỏ" -> táo
    for k in sorted(LEXICON, key=len, reverse=True):
        if len(k) >= 2 and (t.startswith(k + " ") or t.endswith(" " + k)):
            e = LEXICON[k]
            div = e.divisible if qt is None or qt == e.qtype else (qt not in INTEGER_QTYPES)
            return Unit(lemma=t, classifier=e.classifier, divisible=div, plausible=e.plausible)
    if qt is None:
        return Unit(lemma=t, divisible=True)
    return Unit(lemma=t, divisible=qt not in INTEGER_QTYPES and qt != QType.COUNT,
                plausible=_BIG if qt == QType.COUNT else None)


def default_qtype(unit: Unit) -> QType:
    if unit.per:
        return QType.RATE
    e = lookup(unit.lemma)
    if e is not None:
        return e.qtype
    if unit.classifier or not unit.divisible:
        return QType.COUNT
    return QType.ABSTRACT


def unit_class(unit: Unit, qtype: QType) -> str:
    """Coarse class for template lookup: count/person/vnd/usd/mass/time/... ."""
    if qtype == QType.RATE or unit.per:
        return "rate"
    e = lookup(unit.lemma)
    if e is not None:
        return e.cls
    if qtype == QType.MONEY:
        return "usd"
    if qtype == QType.COUNT:
        return "count"
    return qtype.value


def requires_integer(qtype: QType | None, unit: Unit | None) -> bool:
    """Integrality at every node: COUNT/AGE, or any non-rate quantity whose noun is indivisible."""
    if qtype in INTEGER_QTYPES:
        return True
    if qtype in (QType.RATE, QType.RATIO, QType.PERCENT, QType.ABSTRACT, QType.MEASURE, QType.TIME_POINT):
        return False
    return unit is not None and not unit.divisible and unit.per is None


def requires_nonneg(qtype: QType | None) -> bool:
    return qtype in NONNEG_QTYPES


def grid_for(qtype: QType, unit: Unit, original: Fraction) -> Fraction:
    """Resampling grid (T1 §2.5): COUNT 1, VND 500, USD 0.01, PERCENT 5, MEASURE 0.5, AGE 1.
    The grid is coarsened to the original value's own precision (an integer price stays an integer)."""
    original = Fraction(original)
    cls = unit_class(unit, qtype)
    if qtype in (QType.COUNT, QType.AGE) or requires_integer(qtype, unit):
        g = _F(1)
    elif qtype == QType.MONEY:
        g = {"vnd": _F(500), "vnd_k": _F(1, 2), "vnd_m": _F(1, 10), "cent": _F(1)}.get(cls, _F(1, 100))
    elif qtype == QType.PERCENT:
        g = _F(5) if original.denominator == 1 and original % 5 == 0 else _F(1)
    elif qtype == QType.MEASURE:
        g = _F(1, 2)
    elif qtype == QType.RATE:
        g = _F(1, 100)
    else:
        g = _F(1)
    # coarsen to the literal's precision: an integer stays an integer, a multiple of 10 may stay one
    if original.denominator == 1 and g < 1:
        g = _F(1)
    if qtype == QType.MONEY and cls == "vnd" and original % 1000 == 0:
        g = max(g, _F(1000))
    return g
