import json
import re
from pyrsistent import (
    v,
    m,
    s,
    PVector,
    PMap,
    PSet,
    thaw,
)

from pathlib import Path

try:
    module_dir = Path(__file__).parent
except NameError:
    module_dir = Path(".").parent

project_root = module_dir.parent.parent

with open(project_root / "resources/modality-patterns.json") as f:
    modality_patterns = json.loads(f.read())


def split_defs(s: str) -> set[str]:
    comma_splits = s.split("、")
    paren_splits = s.split("（")
    defs = (
        [re.sub(r"[（）]", "", d) for d in paren_splits]
        + comma_splits
        + [re.sub(r"[（）]", "", ds) for d in comma_splits for ds in d.split("（")]
    )
    return set(defs)


def termpp(s):
    """Returns the first pattern name."""
    m = re.search(r"（([^）]+)）", s)
    if m:
        return m.group(1)
    else:
        return s.split("、")[0]


# TODO ものの as 文中接続表現
with open(project_root / "resources/connectives-patterns.json") as f:
    connectives_patterns = json.loads(f.read())
    connectives_classifications = {
        sub_pattern: pattern["kinou"]
        for pattern in connectives_patterns
        for sub_pattern in split_defs(pattern["conjunction"])
    }

with open(project_root / "resources/connectives-regexes.json") as f:
    connectives_regexes = json.loads(f.read())
    connectives_classifications = connectives_classifications | {
        sub_pattern: pattern["kinou"]
        for pattern in connectives_regexes
        for sub_pattern in split_defs(pattern["conjunction"])
    }


def parallel_expand(
    element: PVector | PMap | PSet,
) -> PVector[PVector[PMap[str, str | PMap[str, str]]]]:
    """
    Recursively walks a nested structure of PVector, PSet, and PMap objects,
    building a vector of Spacy token patterns.

    :param element: The nested structure to process.
    :return: A nested PVector of Spacy token patterns.
    """
    if isinstance(element, PMap):
        # A single token pattern, wrap in a vector
        return v(v(element))
    elif isinstance(element, PVector):
        # Initialize the result as an empty vector for concatenation
        result = v(v())
        for item in element:
            expanded = parallel_expand(item)
            # Concatenate each expanded item with each vector in the current result
            new_result = v()
            if result == v(v()):
                # If result is still an empty vector of vectors, replace it directly
                new_result = expanded
            else:
                for res_item in result:
                    for exp_item in expanded:
                        new_result = new_result.append(res_item + exp_item)
            result = new_result
        return result
    elif isinstance(element, PSet):
        # Process a set (OR semantics)
        result = v()
        for item in element:
            expanded = parallel_expand(item)
            # For each item in the set, create new branches
            for exp_item in expanded:
                result = result.append(exp_item)
        return result
    else:
        raise TypeError(f"Unsupported type: {type(element)} for {element}")


def expand_patterns(patterns):
    """Expands the DSL patterns into vanilla Python data to be passed to
    spaCy's Matcher."""
    return {
        pattern_name: d | {"pattern": thaw(parallel_expand(d["pattern"]))}
        for pattern_name, d in patterns.items()
    }


# Mini-DSL for patterns

AND = v
OR = s
# keep m as it is a good mnemonic for morpheme


NAI = AND(
    m(ORTH=m(IN=AND("ない", "無い", "ず", "ぬ", "なかっ", "無かっ", "ねえ"))),
    m(ORTH="た", OP="?"),
)

MASEN = OR(
    AND(m(ORTH="ませ"), m(ORTH="ん")),
    AND(m(ORTH="ませ"), m(ORTH="ん"), m(ORTH="でし"), m(ORTH="た")),
)

NAI_MASEN = MASEN.add(NAI)

ARU = OR(
    AND(
        m(
            LEMMA=m(IN=AND("ある", "有る")),
            ORTH=m(NOT_IN=AND("あろう", "有ろう", "あれ", "有れ")),
        ),
        m(ORTH="た", OP="?"),
    ),
    AND(m(NORM="有る"), m(ORTH="ます")),
    AND(m(NORM="有る"), m(ORTH="まし"), m(ORTH="た")),
)

ARIMASEN = OR(
    AND(m(LEMMA=m(IN=AND("有る", "ある"))), m(ORTH="ませ"), m(ORTH="ん")),
    AND(
        m(LEMMA=m(IN=AND("有る", "ある"))),
        m(ORTH="ませ"),
        m(ORTH="ん"),
        m(ORTH="でし"),
        m(ORTH="た"),
    ),
)

DEARU = AND(m(ORTH="で"), ARU)

DA = OR(
    AND(
        m(ORTH=m(IN=AND("だ", "です", "でし", "だっ")), NORM=m(NOT_IN=AND("た"))),
        m(ORTH="た", OP="?"),
    ),
    DEARU,
)

DEWA = OR(AND(m(ORTH="で"), m(ORTH="は")), AND(m(ORTH="じゃ")))

VERB_TE = m(ORTH=m(IN=AND("て", "で")))

VERB_TEIRU = AND(VERB_TE, m(NORM="居る"))

DAROU_STRICT = m(ORTH=m(IN=AND("だろう", "だろ")))

DEAROU = AND(m(ORTH="で"), m(ORTH="あろう"))

DAROU = OR(
    AND(m(ORTH=m(IN=AND("だろう", "でしょう", "だろ", "たろう")))),
    DEAROU,
)


NOUN_NEGATION = AND(DEWA, OR(NAI, ARIMASEN))

KOTO = m(NORM=m(IN=AND("こと", "事")))

RARE = m(NORM="られる")
RERU = m(NORM="れる")

SURU = m(NORM="為る")
NARU = m(NORM=m(IN=AND("成る", "なる")), ORTH=m(NOT_IN=AND("なろう")))

# Covers parsing errors:
TOMO = OR(
    AND(
        m(ORTH="と"),
        m(ORTH="も"),
    ),
    m(NORM="共"),
)


VERB_PHRASE = OR(
    AND(
        m(TAG=m(IN=AND("動詞-非自立可能", "動詞-一般"))),
        m(POS="AUX", OP="*", NORM=m(NOT_IN=AND("ない", "無い"))),
    ),
    AND(
        m(TAG="名詞-普通名詞-サ変可能"),
        OR(
            m(
                POS="AUX", OP="+", NORM=m(NOT_IN=AND("できる", "出来る"))
            ),  # TODO check * <> +
            SURU,
        ),
    ),
    AND(SURU, m(POS="AUX", OP="*")),
)

modality_patterns_2 = {
    "だ": {
        "category": ["3認識", "断定"],
        "examples": ["駄目だった", "これは本だ", "この研究は有益だった。"],
        "pattern": DA,
    },
    "のだ": {
        "category": ["4説明", "説明"],
        "examples": ["私のペンなのだ", "彼の成功は努力の結果なのだ。"],
        "pattern": AND(m(NORM="の"), DA),
    },
    "という": {
        "category": [],
        "examples": ["発掘されたという"],
        "pattern": AND(m(ORTH="と"), m(LEMMA=m(IN=AND("言う", "いう")))),
    },
    "だろう": {
        "category": [],
        "examples": ["明日は雨だろう", "対比であろう"],
        "pattern": AND(DAROU),
    },
    "となる": {
        "category": [],
        "examples": ["最終的な結果となる"],
        "pattern": AND(m(ORTH="と"), m(LEMMA=m(IN=AND("成る", "なる")))),
    },
    "とする": {
        "category": [],
        "examples": ["これを基準とする"],
        "pattern": AND(m(ORTH="と"), m(LEMMA=m(IN=AND("為る", "する")))),
    },
    "ものだ": {
        "category": [],
        "examples": ["避けられないものだ"],
        "pattern": AND(m(NORM=m(IN=AND("物", "もの"))), DA),
    },
    "ことができる": {
        "category": [],
        "examples": ["解決することができる"],
        "pattern": AND(KOTO, m(ORTH="が"), m(NORM="出来る")),
    },
    "と思う": {
        "category": [],
        "examples": ["防げると思う"],
        "pattern": AND(m(ORTH="と"), m(NORM="思う", MORPH=m(REGEX="連用形|終止形"))),
    },
    "ことになる": {
        "category": [],
        "examples": ["ことになる"],
        "pattern": AND(KOTO, m(ORTH="に"), m(LEMMA=m(IN=AND("成る", "なる")))),
    },
    "ことだ": {
        "category": [],
        "examples": ["解決することだ"],
        "pattern": AND(KOTO, DA),
    },
    "ではない": {
        "category": [],
        "examples": ["偶然ではない"],
        "pattern": NOUN_NEGATION,
    },
    "のか": {
        "category": [],
        "examples": ["偶然なのか"],
        "pattern": AND(m(NORM="の"), m(ORTH="か")),
    },
    "たい": {
        "category": [],
        "examples": ["検討されたい"],
        "pattern": AND(m(ORTH="たい", POS="AUX")),
    },
    "からだ": {
        "category": [],
        "examples": ["成功したからだ"],
        "pattern": AND(m(ORTH="から"), DA),
    },
    "ようになる": {
        "category": [],
        "examples": ["できるようになる"],
        "pattern": AND(
            m(ORTH="よう"), m(ORTH="に"), m(LEMMA=m(IN=AND("成る", "なる")))
        ),
    },
    "と考えられる": {
        "category": [],
        "examples": ["適切であると考えられる"],
        "pattern": AND(m(ORTH="と"), m(NORM="考える"), m(NORM="られる")),
    },
    "ようだ": {
        "category": [],
        "examples": ["適切のようだ"],
        "pattern": AND(m(ORTH="よう"), DA),
    },
    "にする": {
        "category": [],
        "examples": ["目標にする"],
        "pattern": AND(m(NORM="に"), m(NORM="為る")),
    },
    "なければならない": {
        "category": [],
        "examples": ["努力しなければなりません"],
        "pattern": AND(
            m(ORTH="なけれ"), m(ORTH="ば"), m(ORTH=m(IN=AND("なら", "なり"))), NAI_MASEN
        ),
    },
    "のだろう": {
        "category": [],
        "examples": ["偶然なのだろう"],
        "pattern": AND(m(NORM="の"), DAROU),
    },
    "といえる": {
        "category": [],
        "examples": ["努力の結果と言える"],
        "pattern": AND(m(ORTH="と"), m(LEMMA=m(IN=AND("言える", "いえる")))),
    },
    "かもしれない": {
        "category": [],
        "examples": ["誤差によるものかもしれない"],
        "pattern": AND(m(ORTH="か"), m(ORTH="も"), m(NORM="知れる"), NAI_MASEN),
    },
    "とした": {
        "category": [],
        "examples": ["目標とした"],
        "pattern": AND(m(ORTH="と"), m(ORTH="し"), m(ORTH="た")),
    },
    "必要がある": {
        "category": [],
        "examples": ["採掘する必要がある"],
        "pattern": AND(m(NORM="必要"), m(ORTH="が"), ARU),
    },
    "のだろうか": {
        "category": [],
        "examples": ["誤差なのだろうか"],
        "pattern": AND(m(NORM="の"), DAROU, m(ORTH="か")),
    },
    "わけだ": {
        "category": [],
        "examples": ["成功したわけだ"],
        "pattern": AND(m(ORTH=m(IN=AND("わけ", "訳"))), DA),
    },
    "ことがある": {
        "category": [],
        "examples": ["誤作動することがある"],
        "pattern": AND(KOTO, m(ORTH="が"), ARU),
    },
    "そうだ": {
        "category": [],
        "examples": ["検討するそうだ"],
        "pattern": AND(m(MORPH=m(REGEX="^((?!連用形-一般).)*$")), m(ORTH="そう"), DA),
    },
    "ということだ": {
        "category": [],
        "examples": ["低い結果だということだ"],
        "pattern": AND(m(ORTH="と"), m(LEMMA=m(IN=AND("言う", "いう"))), KOTO, DA),
    },
    "だろうか": {
        "category": ["1表現類型", "疑問"],
        "examples": ["解決するだろうか"],
        "pattern": AND(DAROU, m(ORTH="か")),
    },
    "はずだ": {
        "category": [],
        "examples": ["成功するはずだ"],
        "pattern": AND(m(NORM="筈"), DA),
    },
    "らしい": {
        "category": [],
        "examples": ["成功するらしい"],
        "pattern": AND(m(ORTH="らしい")),
    },
    "ことにする": {
        "category": [],
        "examples": ["再考することにする"],
        "pattern": AND(m(NORM="こと"), m(ORTH="に"), m(LEMMA="する")),
    },
    "てよい": {
        "category": [],
        "examples": ["設置してよい"],
        "pattern": AND(VERB_TE, m(NORM="良い")),
    },
    "と思われる": {
        "category": [],
        "examples": ["その原因と思われる"],
        "pattern": AND(m(ORTH="と"), m(NORM="思う"), m(NORM="れる")),
    },
    "わけではない": {
        "category": [],
        "examples": ["変更できないわけではありません"],
        "pattern": AND(m(ORTH=m(IN=AND("わけ", "訳"))), NOUN_NEGATION),
    },
    "ことはない": {
        "category": [],
        "examples": ["外れることはない"],
        "pattern": AND(KOTO, m(ORTH="は"), OR(NAI, AND(m(NORM="有る"), MASEN))),
    },
    "ではないか": {
        "category": [],
        "examples": ["その結果ではないか"],
        "pattern": AND(NOUN_NEGATION, m(ORTH="か")),
    },
    "なさい": {
        "category": [],
        "examples": ["論じなさい"],
        "pattern": AND(m(NORM="為さる")),
    },
    "のに": {
        "category": [],
        "examples": ["解決できたのに"],
        "pattern": AND(m(ORTH="の"), m(ORTH="に")),
    },
    "のではない": {
        "category": [],
        "examples": ["最善の方法なのではない"],
        "pattern": AND(m(NORM="の"), NOUN_NEGATION),
    },
    "うる": {
        "category": [],
        "examples": ["有効な方法でありうる"],
        "pattern": AND(m(POS="VERB"), m(NORM="得る")),
    },
    "べきだ": {
        "category": [],
        "examples": ["新しい方法を開発すべきだ"],
        "pattern": AND(m(ORTH="べき"), DA),
    },
    "に違いない": {
        "category": [],
        "examples": ["有効な方法に違いない"],
        "pattern": AND(
            m(ORTH="に"), m(NORM="違い"), OR(NAI, AND(m(NORM="有る"), MASEN))
        ),
    },
    "てはならない": {
        "category": [],
        "examples": ["損害する方法を開発してはならない"],
        "pattern": AND(VERB_TE, m(ORTH="は"), m(NORM="成る"), NAI_MASEN),
    },
    "のではないか": {
        "category": [],
        "examples": ["最善の方法なのではないか"],
        "pattern": AND(m(NORM="の"), NOUN_NEGATION, m(ORTH="か")),
    },
    "ものではない": {
        "category": [],
        "examples": ["簡易なものではない"],
        "pattern": AND(m(NORM="物"), NOUN_NEGATION),
    },
    "ことだろう": {
        "category": [],
        "examples": ["有効な方法であることだろう"],
        "pattern": AND(KOTO, DAROU),
    },
    "とされている": {
        "category": [],
        "examples": ["有効な方法とされている"],
        "pattern": AND(
            m(ORTH="と"), m(ORTH="さ"), m(NORM="れる"), VERB_TE, m(NORM="居る")
        ),
    },
    "のではないだろうか": {
        "category": [],
        "examples": ["最善の方法なのではないだろうか"],
        "pattern": AND(m(NORM="の"), NOUN_NEGATION, DAROU, m(ORTH="か")),
    },
    "ものとする": {
        "category": [],
        "examples": ["乙のものとする"],
        "pattern": AND(m(NORM="物"), m(ORTH="と"), m(NORM="為る")),
    },
    "ないか": {
        "category": [],
        "examples": ["決定しないか"],
        "pattern": AND(NAI, m(ORTH="か")),
    },
    "てほしい": {
        "category": [],
        "examples": ["検討してほしい"],
        "pattern": AND(VERB_TE, m(NORM="欲しい")),
    },
    "でもない": {
        "category": [],
        "examples": ["最善の方法でもない"],
        "pattern": AND(m(ORTH="で"), m(ORTH="も"), OR(NAI, AND(m(NORM="有る"), MASEN))),
    },
    "にすぎない": {
        "category": [],
        "examples": ["推測に過ぎない"],
        "pattern": AND(m(ORTH="に"), m(NORM="過ぎる"), NAI_MASEN),
    },
    "が考えられる": {
        "category": [],
        "examples": ["方法が考えられる"],
        "pattern": AND(m(ORTH="が"), m(NORM="考える"), m(NORM="られる")),
    },
    "まい": {
        "category": [],
        "examples": ["失敗するまい"],
        "pattern": AND(m(POS="AUX", ORTH="まい")),
    },
    "といわれている": {
        "category": [],
        "examples": ["有効な方法と言われている"],
        "pattern": AND(
            m(ORTH="と"), m(NORM="言う"), m(NORM="れる"), VERB_TE, m(NORM="居る")
        ),
    },
    "ざるをえない": {
        "category": [],
        "examples": ["開発せざるをえない"],
        "pattern": AND(m(ORTH="ざる"), m(ORTH="を"), m(NORM="得る"), NAI_MASEN),
    },
    "ではないだろうか": {
        "category": [],
        "examples": ["最善の方法ではないだろうか"],
        "pattern": AND(
            m(ORTH="で"),
            m(ORTH="は"),
            OR(m(ORTH="なかろう"), AND(NAI, DAROU)),
            m(ORTH="か"),
        ),
    },
    "てしまう": {
        "category": ["考えてしまった"],
        "examples": ["新しい方法を開発してしまう"],
        "pattern": AND(VERB_TE, m(NORM="仕舞う")),
    },
    "（よ）う": {
        "category": ["1表現類型", "勧誘"],
        "examples": ["行こう", "新しい方法を開発しよう"],
        "pattern": AND(m(POS=m(IN=AND("VERB", "AUX")), MORPH=m(REGEX="意志推量形"))),
    },
}

sf_definitions = {
    "があげられる": {
        "category": ["認識", "あげる"],
        "examples": ["参考例があげられる", "事項が挙げられる"],
        "pattern": AND(m(ORTH="が"), m(NORM=m(IN=AND("上げる", "挙げる"))), RARE),
    },
    "Nにある": {
        "category": ["断定・存在", "ある"],
        "examples": ["可能性にある"],
        "pattern": AND(
            m(
                POS="NOUN", NORM=m(NOT_IN=AND("こと", "事"))
            ),  # NOTE Do not match on KOTO, as that is another pattern.
            m(ORTH="に"),
            m(NORM="有る"),
        ),
    },
    "可能性がある": {
        "category": ["断定・存在", "ある"],
        "examples": ["可能性がある"],
        "pattern": AND(m(NORM="可能性"), m(ORTH="が"), m(NORM="有る")),
    },
    "ことがある": {
        "category": ["断定・存在", "ある"],
        "examples": ["ことがある"],
        "pattern": AND(KOTO, m(ORTH="が"), m(NORM="有る")),
    },
    "ことにある": {
        "category": ["断定・存在", "ある"],
        "examples": ["ことにある"],
        "pattern": AND(KOTO, m(ORTH="に"), m(NORM="有る")),
    },
    "点がある": {
        "category": ["断定・存在", "ある"],
        "examples": ["点がある"],
        "pattern": AND(m(NORM="点"), m(ORTH="が"), m(NORM="有る")),
    },
    "場合がある": {
        "category": ["断定・存在", "ある"],
        "examples": ["場合がある"],
        "pattern": AND(m(NORM="場合"), m(ORTH="が"), m(NORM="有る")),
    },
    "必要がある": {
        "category": ["断定・存在", "ある"],
        "examples": ["採掘する必要がある"],
        "pattern": AND(m(NORM="必要"), m(ORTH="が"), ARU),
    },
    "ものがある": {
        "category": ["断定・存在", "ある"],
        "examples": ["採掘するものがある"],
        "pattern": AND(m(NORM="物"), m(ORTH="が"), ARU),
    },
    "Vという": {
        "category": ["推量・意志", "いう"],
        "examples": ["解決されるという", "終わるという"],
        "pattern": AND(VERB_PHRASE, m(ORTH="と"), m(NORM="言う")),
    },
    "といえる": {
        "category": ["可能（性）", "いう"],
        "examples": ["そうといえる", "妥当だといえる"],
        "pattern": AND(
            m(ORTH="と"),
            m(NORM=m(IN=["言える", "いえる"])),
        ),
    },
    "Vとはいえない": {
        "category": ["推量・意志", "いう"],
        "examples": ["解決したとはいえない"],
        "pattern": AND(
            VERB_PHRASE,
            m(ORTH="と"),
            m(ORTH="は"),
            m(NORM=m(IN=["言える", "いえる"])),
            NAI,
        ),
    },
    "ことを意図する": {
        "category": ["意志・措置行為", "意図する"],
        "examples": ["何をすることを意図するのか"],
        "pattern": AND(
            KOTO,
            m(ORTH="を"),
            m(NORM="意図"),
            SURU,
        ),
    },
    "ことを意味する": {
        "category": ["意志・措置行為", "意味する"],
        "examples": ["重大なことを意味する"],
        "pattern": AND(
            KOTO,
            m(ORTH="を"),
            m(NORM="意味"),
            SURU,
        ),
    },
    "Vよう": {
        "category": ["推量・意志", "う"],
        "examples": ["解決策を試みよう"],
        "pattern": AND(
            m(
                POS=m(IN=AND("VERB", "AUX")),
                MORPH=m(REGEX="意志推量形"),
            )
        ),
    },
    "Vだろう": {
        "category": ["推量・意志", "う"],
        "examples": ["来るだろう"],
        "pattern": AND(
            VERB_PHRASE,
            DAROU_STRICT,
        ),
    },
    "であろう": {
        "category": ["推量・意志", "う"],
        "examples": ["そうであろう", "可能性が高いであろう"],
        "pattern": DEAROU,
    },
    "となろう": {
        "category": ["推量・意志", "う"],
        "examples": ["そうとなろう", "結果となろう"],
        "pattern": AND(
            m(ORTH="と"),
            m(
                LEMMA=m(IN=AND("成る", "なる")),
                MORPH=m(REGEX="意志推量形"),
            ),
        ),
    },
    "ないだろう": {  # NOTE: Vだろう
        "category": ["推量・意志", "う"],
        "examples": ["雨は降らないだろう", "問題はないだろう"],
        "pattern": AND(NAI, DAROU),
    },
    "Vうる": {
        "category": ["推量・意志", "うる"],
        "examples": ["ありうる"],
        "pattern": AND(
            m(POS="VERB"),
            m(NORM="得る"),
        ),
    },
    "ことが多い": {
        "category": ["認識", "多い"],
        "examples": ["遅延することが多い"],
        "pattern": AND(
            KOTO,
            m(ORTH="が"),
            m(NORM="多い"),
        ),
    },
    "と思う": {
        "category": ["推量・意志", "思う"],
        "examples": ["勝つと思う", "ベストだと思う"],
        # "negative_examples": ["と思われる"],
        "pattern": AND(
            m(ORTH="と"),
            m(NORM="思う"),
        ),
    },
    "と思われる": {
        "category": ["推量・意志", "思う"],
        "examples": ["成功すると思われる"],
        "pattern": AND(
            m(ORTH="と"),
            m(NORM="思う"),
            RERU,
        ),
    },
    "ように思われる": {
        "category": ["推量・意志", "思う"],
        "examples": ["彼は疲れているように思われる", "変化がないように思われる"],
        "pattern": AND(
            m(NORM="よう"),
            m(ORTH="に"),
            m(NORM="思う"),
            m(NORM="れる"),
        ),
    },
    "Vのか": {
        "category": ["疑問", "か"],
        "examples": ["解決するのか"],
        "pattern": AND(
            VERB_PHRASE,
            m(ORTH="の"),
            m(ORTH="か"),
        ),
    },
    "Vだろうか": {
        "category": ["疑問", "か"],
        "examples": ["それが起こるだろうか"],
        "pattern": AND(
            VERB_PHRASE,
            DAROU_STRICT,
            m(ORTH="か"),
        ),
    },
    "ないだろうか": {  # NOTE: Vだろうか
        "category": ["疑問", "か"],
        "examples": ["雨は降らないだろうか", "問題はないだろうか"],
        "pattern": AND(
            NAI,
            DAROU,
            m(ORTH="か"),
        ),
    },
    "であろうか": {
        "category": ["疑問", "か"],
        "examples": ["それが可能であろうか", "解決策が見つかるであろうか"],
        "pattern": AND(
            DEAROU,
            m(ORTH="か"),
        ),
    },
    "ADJだろうか": {
        "category": ["疑問", "か"],
        "examples": ["大きいだろうか"],
        "pattern": AND(
            m(POS="ADJ"),
            DAROU_STRICT,
            m(ORTH="か"),
        ),
    },
    "のだろうか": {
        "category": ["疑問", "か"],
        "examples": ["それが真実なのだろうか", "彼が正しいのだろうか"],
        "pattern": AND(
            m(ORTH="の"),
            DAROU,
            m(ORTH="か"),
        ),
    },
    "Vのではなかろうか": {
        "category": ["疑問", "か"],
        "examples": ["解決するのではなかろうか"],
        "pattern": AND(
            VERB_PHRASE,
            m(ORTH="の"),
            m(ORTH="で"),
            m(ORTH="は"),
            m(ORTH="なかろう"),
            m(ORTH="か"),
        ),
    },
    "べきではないか": {
        "category": ["疑問", "か"],
        "examples": ["行くべきではないか"],
        "pattern": AND(
            m(ORTH="べき"),
            m(ORTH="で"),
            m(ORTH="は"),
            NAI,
            m(ORTH="か"),
        ),
    },
    "Nなのか": {
        "category": ["疑問", "か"],
        "examples": ["それが問題なのか"],
        "pattern": AND(
            m(POS="NOUN"),
            m(ORTH="な"),
            m(ORTH="の"),
            m(ORTH="か"),
        ),
    },
    "とは限らない": {
        "category": ["否定（部分否定）", "限らない"],
        "examples": ["成功するとは限らない", "簡単だとはかぎらない"],
        "pattern": AND(
            m(ORTH="と"),
            m(ORTH="は"),
            m(NORM="限る"),
            NAI,
        ),
    },
    "が確認される": {
        "category": ["認識", "確認する"],
        "examples": ["その事実が確認される"],
        "pattern": AND(
            m(ORTH="が"),
            m(NORM="確認"),
            SURU,
            RERU,
        ),
    },
    "Vことを確認する": {
        "category": ["認識", "確認する"],
        "examples": ["それをすることを確認する"],
        "pattern": AND(
            VERB_PHRASE,
            KOTO,
            m(ORTH="を"),
            m(NORM="確認"),
            SURU,
        ),
    },
    "と仮定する": {
        "category": ["仮の措置", "仮定する"],  # changed from "意志・措置行為"
        "examples": ["それが真実だと仮定する"],
        "pattern": AND(
            m(ORTH="と"),
            m(NORM="仮定"),
            SURU,
        ),
    },
    "Vかもしれない": {
        "category": ["可能（性）", "かもしれない"],
        "examples": ["雨が降るかもしれない"],
        "pattern": AND(
            VERB_PHRASE,
            m(ORTH="か"),
            m(ORTH="も"),
            m(NORM="知れる"),
            NAI_MASEN,
        ),
    },
    "ばADJかもしれない": {
        "category": ["可能（性）", "かもしれない"],
        "examples": ["高ければ良いかもしれない", "簡単であれば便利かもしれない"],
        "pattern": AND(
            m(ORTH="ば"),
            m(POS=m(IN=AND("ADJ", "AUX"))),
            m(ORTH="か"),
            m(ORTH="も"),
            m(NORM="知れる"),
            NAI_MASEN,
        ),
    },
    "が考えられる": {
        "category": ["推量・意志", "考える"],
        "examples": ["その理由が考えられる", "工事を中止することが考えられる"],
        "pattern": AND(
            m(ORTH="が"),
            m(NORM="考える"),
            RARE,
        ),
    },
    "考えてみよう": {
        "category": ["推量・意志", "考える"],
        "examples": ["その提案を考えてみよう", "別の方法を考えてみよう"],
        "pattern": AND(
            m(NORM="考える"),
            VERB_TE,
            m(NORM="見る", MORPH=m(REGEX="意志推量形")),
        ),
    },
    "と考えられる": {
        "category": ["推量・意志", "考える"],
        "examples": ["それが最善と考えられる", "その計画は成功すると考えられる"],
        "pattern": AND(
            m(ORTH="と"),
            m(NORM="考える"),
            RARE,
        ),
    },
    "と考える": {
        "category": ["推量・意志", "考える"],
        "examples": ["私はそれが可能だと考える", "彼はそうすると考える"],
        "pattern": AND(
            m(ORTH="と"),
            m(NORM="考える"),
        ),
    },
    "が期待できる": {
        "category": ["願望・期待", "期待する"],
        "examples": ["成果が期待できる"],
        "pattern": AND(
            m(ORTH="が"),
            m(NORM="期待"),
            m(NORM="出来る"),
        ),
    },
    "とも期待できる": {
        "category": ["願望・期待", "期待する"],
        "examples": ["大きな成果とも期待できる", "良い結果とも期待できる"],
        "pattern": AND(
            TOMO,
            m(NORM="期待"),
            m(NORM="出来る"),
        ),
    },
    "と期待される": {
        "category": ["願望・期待", "期待する"],
        "examples": ["実現すると期待される"],
        "pattern": AND(
            m(ORTH="と"),
            m(NORM="期待"),
            SURU,
            RERU,
        ),
    },
    "が示唆される": {
        "category": ["認識", "示唆する"],
        "examples": ["新たな道が示唆される"],
        "pattern": AND(
            m(ORTH="が"),
            m(NORM="示唆"),
            SURU,
            RERU,
        ),
    },
    "Vことを示唆する": {
        "category": ["認識", "示唆する"],
        "examples": ["結果が変化することを示唆する"],
        "pattern": AND(
            VERB_PHRASE,
            KOTO,
            m(ORTH="を"),
            m(NORM="示唆"),
            SURU,
        ),
    },
    "Vことが指摘される": {
        "category": ["認識", "指摘する"],
        "examples": ["変動することが指摘される"],
        "pattern": AND(
            VERB_PHRASE,
            KOTO,
            m(ORTH="が"),
            m(NORM="指摘"),
            SURU,
            RERU,
        ),
    },
    "ADJことが指摘される": {
        "category": ["認識", "指摘する"],
        "examples": ["重要なことが指摘される", "必要なことが指摘される"],
        "pattern": AND(
            m(POS="ADJ"),
            m(POS="AUX", OP="?"),
            KOTO,
            m(ORTH="が"),
            m(NORM="指摘"),
            SURU,
            RERU,
        ),
    },
    "Vと指摘する": {
        "category": ["意志・措置行為", "指摘する"],
        "examples": ["専門家が活発化すると指摘する"],
        "pattern": AND(
            VERB_PHRASE,
            m(ORTH="と"),
            m(NORM="指摘"),
            SURU,
        ),
    },
    "ADJことを指摘する": {  # TODO: check ADJ=na+i
        "category": ["意志・措置行為", "指摘する"],
        "examples": ["研究者が必要なことを指摘する"],
        "pattern": AND(
            m(POS="ADJ"),
            m(POS="AUX", OP="?"),
            KOTO,
            m(ORTH="を"),
            m(NORM="指摘"),
            SURU,
        ),
    },
    "が示される": {
        "category": ["認識", "示す"],
        "examples": ["結果が示される"],
        "pattern": AND(
            m(ORTH="が"),
            m(NORM="示す"),
            RERU,
        ),
    },
    "Vことを示す": {
        "category": ["意志・措置行為", "示す"],
        "examples": ["深刻化することを示す"],
        "pattern": AND(
            VERB_PHRASE,
            KOTO,
            m(ORTH="を"),
            m(NORM="示す"),
        ),
    },
    "ことが知られている": {
        "category": ["認識", "知る"],
        "examples": ["未解決なことが知られている"],
        "pattern": AND(
            KOTO,
            m(ORTH="が"),
            m(NORM="知る"),
            RERU,
            VERB_TE,
            m(NORM="居る"),
        ),
    },
    "Vことが推測される": {
        "category": ["推量・意志", "推測する"],
        "examples": ["することが推測される"],
        "pattern": AND(
            VERB_PHRASE,
            KOTO,
            m(ORTH="が"),
            m(NORM="推測"),
            SURU,
            RERU,
        ),
    },
    "ものと推測される": {
        "category": ["推量・意志", "推測する"],
        "examples": ["存在しないものと推測される"],
        "pattern": AND(
            m(NORM="物"),
            m(ORTH="と"),
            m(NORM="推測"),
            SURU,
            RERU,
        ),
    },
    "Vことが推定される": {
        "category": ["推量・意志", "推定する"],
        "examples": ["その計画が成功することが推定される"],
        "pattern": AND(
            VERB_PHRASE,
            KOTO,
            m(ORTH="が"),
            m(NORM="推定"),
            SURU,
            RERU,
        ),
    },
    "Vたい": {
        "category": ["願望・期待", "たい"],
        "examples": ["検討されたい", "参照されたい"],
        "pattern": AND(
            m(POS=m(IN=AND("VERB", "AUX")), OP="+"),  # TODO: VERB_PHRASE?
            m(ORTH="たい", POS="AUX"),
        ),
    },
    "ADJである": {
        "category": ["断定・存在", "である"],
        "examples": ["簡単である"],
        "pattern": AND(
            m(POS="ADJ", NORM=m(NOT_IN=AND("必要"))),
            DEARU,
        ),
    },
    "ADJそうである": {
        "category": ["断定・存在", "である"],
        # TODO: check
        "examples": ["それは可能そうである"],
        "pattern": AND(
            m(POS="ADJ"),
            m(POS="AUX", OP="?"),
            m(NORM="そう"),
            DEARU,
        ),
    },
    "Nである": {  # NOTE: ことである, ためである, ...
        "category": ["断定・存在", "である"],
        "examples": ["これは事実である"],
        "pattern": AND(
            m(
                POS="NOUN",
                NORM=m(
                    NOT_IN=AND(
                        "こと", "事", "ため", "為", "もの", "物", "ゆえん", "所以"
                    )
                ),
            ),
            DEARU,
        ),
    },
    "Nなのである": {
        "category": ["断定・存在", "である"],
        "examples": ["彼がリーダーなのである", "それが理由なのである"],
        "pattern": AND(
            m(POS="NOUN"),
            m(ORTH="な"),
            m(ORTH="の"),
            DEARU,
        ),
    },
    "ADJなのである": {
        "category": ["断定・存在", "である"],
        "examples": ["重要なのである", "必要なのである"],
        "pattern": AND(
            m(POS="ADJ"),
            m(ORTH="な"),
            m(ORTH="の"),
            DEARU,
        ),
    },
    "Vのである": {
        "category": ["断定・存在", "である"],
        "examples": ["出発するのである", "勉強するのである"],
        "negative_examples": ["それが結論のである"],
        "pattern": AND(
            VERB_PHRASE,
            m(ORTH="の"),
            DEARU,
        ),
    },
    "からである": {
        "category": ["断定・存在", "である"],
        "examples": ["経験したからである", "研究からである"],
        "pattern": AND(
            m(ORTH="から", POS=m(IN=("ADP", "SCONJ"))),
            DEARU,
        ),
    },
    "ことである": {
        "category": ["断定・存在", "である"],
        "examples": ["大切なことである", "注意すべきことである"],
        "pattern": AND(KOTO, DEARU),
    },
    "ためである": {
        "category": ["断定・存在", "である"],
        "examples": ["安全のためである"],
        "pattern": AND(m(ORTH="ため"), DEARU),
    },
    "のである": {
        "category": ["断定・存在", "である"],
        "examples": ["それが結論のである", "私の意見のである"],
        "pattern": AND(
            m(ORTH="の"),
            DEARU,
        ),
    },
    "必要である": {
        "category": ["断定・存在", "である"],
        "examples": ["勉強が必要である", "休息が必要である"],
        "pattern": AND(
            m(NORM="必要"),
            DEARU,
        ),
    },
    "ものである": {
        "category": ["断定・存在", "である"],
        "examples": ["それは貴重なものである"],
        "pattern": AND(
            m(NORM="物"),
            DEARU,
        ),
    },
    "ゆえんである": {
        "category": ["断定・存在", "である"],
        "examples": ["彼の成功のゆえんである", "この問題のゆえんである"],
        "pattern": AND(
            m(NORM=m(IN=AND("ゆえん", "所以"))),
            DEARU,
        ),
    },
    "Nができる": {
        "category": ["可能（性）", "できる"],
        "examples": ["仕事ができる"],
        "pattern": AND(
            m(POS="NOUN", NORM=m(NOT_IN=AND("こと", "事"))),
            m(ORTH="が"),
            m(NORM="出来る"),
        ),
    },
    "ことができる": {
        "category": ["可能（性）", "できる"],
        "examples": ["学ぶことができる", "解決することができる"],
        "pattern": AND(
            KOTO,
            m(ORTH="が"),
            m(NORM="出来る"),
        ),
    },
    # TODO: check
    "V（する動詞）できる": {
        "category": ["可能（性）", "できる"],
        "examples": ["変更できる"],  # FIXME
        "pattern": AND(
            m(
                TAG=m(
                    IN=AND(
                        "名詞-普通名詞-サ変可能",
                        "名詞-普通名詞-サ変形状詞可能",
                        "接尾辞-名詞的-サ変可能",
                        "接尾辞-名詞的-サ変形状詞可能",
                    )
                )
            ),
            m(NORM="出来る"),
        ),
    },
    "Vてしまう": {
        "category": ["NA", "てしまう"],
        "examples": ["忘れてしまう"],
        "pattern": AND(
            VERB_PHRASE,
            VERB_TE,
            m(NORM="仕舞う"),
        ),
    },
    "Vてみたい": {
        "category": ["NA", "たい"],  # TODO てみる→たい
        "examples": ["試してみたい"],
        "pattern": AND(
            VERB_PHRASE,
            VERB_TE,
            m(NORM="見る"),
            m(NORM="たい"),
        ),
    },
    "Vてもよい": {
        "category": ["容認", "てよい"],
        "examples": ["食べてもよい"],
        "pattern": AND(
            VERB_PHRASE,
            VERB_TE,
            m(ORTH="も"),
            m(NORM="良い"),
        ),
    },
    "Nとおく": {
        "category": ["仮の措置", "とおく"],  # TODO category
        "examples": ["数字をＸと置く"],
        "pattern": AND(
            m(POS="NOUN"),
            m(ORTH="と"),
            m(NORM=m(IN=AND("おく", "置く"))),
        ),
    },
    "ADJとする": {
        "category": ["意志・措置行為", "とする"],
        "examples": ["重要とする"],
        "pattern": AND(
            m(POS="ADJ"),
            m(ORTH="と"),
            SURU,
        ),
    },
    "Nとする": {
        "category": ["意志・措置行為", "とする"],
        "examples": ["彼をリーダーとする", "これを最優先事項とする"],
        "pattern": AND(
            m(POS="NOUN"),
            m(ORTH="と"),
            SURU,
        ),
    },
    "Vとする": {
        "category": ["意志・措置行為", "とする"],
        "examples": ["異なるとする"],
        "pattern": AND(
            VERB_PHRASE,
            m(ORTH="と"),
            SURU,
        ),
    },
    "ADJとなる": {
        "category": ["認識", "となる"],
        "examples": ["可能となる", "重要となる"],
        "pattern": AND(
            m(POS="ADJ"),
            m(ORTH="と"),
            NARU,
        ),
    },
    "Nとなる": {
        "category": ["認識", "となる"],
        "examples": ["問題となる", "課題となる"],
        "pattern": AND(
            m(POS="NOUN"),
            m(ORTH="と"),
            NARU,
        ),
    },
    "Vこととなる": {
        "category": ["認識", "となる"],
        "examples": ["出発することとなる", "変更することとなる"],
        "pattern": AND(
            VERB_PHRASE,
            KOTO,
            m(ORTH="と"),
            NARU,
        ),
    },
    "疑いない": {
        "category": ["否定（部分否定）", "ない"],
        "examples": ["成功することに疑いない", "彼が来ることに疑いない"],
        "pattern": AND(m(NORM="疑い"), NAI_MASEN),
    },
    "いうまでもない": {
        "category": ["否定（部分否定）", "ない"],
        "examples": ["それはいうまでもない"],
        "pattern": AND(
            m(NORM="言う"),
            m(NORM="まで"),
            m(ORTH="も"),
            NAI,
        ),
    },
    "ことはない": {
        "category": ["否定（部分否定）", "ない"],
        "examples": ["毎日来ることはない", "必ずしもそうすることはない"],
        "pattern": AND(
            KOTO,
            m(ORTH="は"),
            NAI,
        ),
    },
    "na-ADJがない": {
        "category": ["否定（部分否定）", "ない"],
        # TODO: check
        "examples": ["検査する必要がない"],
        "pattern": AND(
            OR(m(POS="ADJ"), m(TAG="名詞-普通名詞-形状詞可能")),
            m(POS="AUX", OP="?"),
            m(ORTH="が"),
            NAI_MASEN,
        ),
    },
    "なければならない": {
        "category": ["否定（部分否定）", "ない"],
        "examples": ["修正しなければならない"],
        "pattern": AND(
            m(ORTH="なけれ"), m(ORTH="ば"), m(ORTH=m(IN=AND("なら", "なり"))), NAI_MASEN
        ),
    },
    "すぎない": {
        "category": ["否定（部分否定）", "ない"],
        "examples": ["過信しすぎない"],
        "pattern": AND(m(NORM="過ぎる"), NAI_MASEN),
    },
    "ものではない": {
        "category": ["否定（部分否定）", "ない"],
        "examples": ["それは許されるものではない", "見逃すものではない"],
        "pattern": AND(m(NORM="物"), m(ORTH="で"), m(ORTH="は"), NAI_MASEN),
    },
    "ADJではない": {
        "category": ["否定（部分否定）", "ない"],
        "examples": ["簡単ではない", "不可能ではない"],
        "pattern": AND(m(POS="ADJ"), m(ORTH="で"), m(ORTH="は"), NAI_MASEN),
    },
    "わけではない": {
        "category": ["否定（部分否定）", "ない"],
        "examples": ["単なるわけではない"],
        "pattern": AND(m(NORM="訳"), m(ORTH="で"), m(ORTH="は"), NAI_MASEN),
    },
    "ことになる": {
        "category": ["認識", "なる"],
        "examples": ["そうすることになる"],
        "pattern": AND(KOTO, m(ORTH="に"), NARU),
    },
    "が望ましい": {
        "category": ["願望・期待", "望ましい"],
        "examples": ["平和が望ましい"],
        "pattern": AND(
            m(ORTH="が"),
            m(NORM="望ましい"),
        ),
    },
    "と報告している": {
        "category": ["認識", "報告する"],
        "examples": ["未解決と報告している"],
        "pattern": AND(
            m(ORTH="と"),
            m(NORM="報告"),
            SURU,
            VERB_TEIRU,
        ),
    },
    "ことがみてとれる": {
        "category": ["NA", "みてとれる"],
        "examples": ["変化することがみてとれる"],
        "pattern": AND(
            KOTO,
            m(ORTH="が"),
            m(NORM="見る"),
            VERB_TE,
            m(NORM="とれる"),
        ),
    },
    "ことを認める": {
        "category": ["NA", "認める"],
        "examples": [
            "そのことを認める",
            "存在することを認める",
        ],
        "pattern": AND(KOTO, m(ORTH="を"), m(NORM="認める")),
    },
    "がよいとする": {  # TODO: 「この方法がよい」が命題になるので，そのあとに「とする」という書き手の判断を述べるという機能です．
        "category": ["NA", "よい"],
        "examples": ["この方法がよいとする"],
        "pattern": AND(
            m(ORTH="が"),
            m(NORM="良い"),
            m(ORTH="と"),
            SURU,
        ),
    },
    "ばよい": {
        "category": ["NA", "よい"],
        "examples": ["早く行けばよい", "そう思えばよい"],
        "pattern": AND(
            m(ORTH="ば"),
            m(NORM="良い"),
        ),
    },
    "ことがADJ予想される": {
        "category": ["推量・意志", "予想する"],
        "examples": ["ことが大きく予想される"],
        "pattern": AND(KOTO, m(ORTH="が"), m(POS="ADJ"), m(NORM="予想"), SURU, RERU),
    },
    "と予測される": {
        "category": ["推量・意志", "予想する"],
        "examples": ["この傾向は続くと予測される"],
        "pattern": AND(
            m(ORTH="と"),
            m(NORM="予測"),
            SURU,
            RERU,
        ),
    },
    "ものと理解されている": {
        "category": ["認識", "理解する"],
        "examples": ["ものと理解されている"],  # これが常識である...
        "pattern": AND(
            m(NORM="物"),
            m(ORTH="と"),
            m(NORM="理解"),
            SURU,
            RERU,
            VERB_TEIRU,
        ),
    },
    "がわかる": {
        "category": ["認識", "わかる"],
        "examples": ["ことがわかる"],  # 事情が深刻である...
        "pattern": AND(
            m(ORTH="が"),
            m(NORM="分かる"),
        ),
    },
}
# {"": {"category": [], "examples": [], "pattern": v()}}

sf_classifications = {sf_name: d["category"] for sf_name, d in sf_definitions.items()}

sf_patterns = expand_patterns(sf_definitions)

if __name__ == "__main__":
    from pprint import pprint
    import pandas as pd

    pd.DataFrame.from_dict(
        {
            sf_name: {
                "大分類": d["category"][0],
                "細分類": d["category"][1],
                "例文": " | ".join(d["examples"]),
            }
            for sf_name, d in sf_definitions.items()
        },
        orient="index",
    ).to_excel("sf_patterns.xlsx")
    pd.DataFrame.from_dict(
        {
            sf_name: {
                "大分類": d["category"][0],
                "細分類": d["category"][1],
                "例文": " | ".join(d["examples"]),
            }
            for sf_name, d in sf_definitions.items()
        },
        orient="index",
    ).to_csv("sf_patterns.csv")

    pprint(modality_patterns)
    pprint(modality_patterns_2)
    pprint(sf_patterns)
    pprint(connectives_patterns)
    pprint(connectives_regexes)
    pprint(connectives_classifications)
