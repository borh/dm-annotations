import json
import re
from pyrsistent import (
    v,
    m,
    s,
    PVector,
    PMap,
    PSet,
)

from pathlib import Path

try:
    module_dir = Path(__file__).parent
except NameError:
    module_dir = Path(".").parent


project_root = module_dir.parent.parent

with open(project_root / "resources/modality-patterns.json") as f:
    modality_patterns = json.loads(f.read())


def split_defs(s: str) -> list[str]:
    defs = s.split("（")
    defs = [re.sub(r"[（）]", "", d) for d in defs]
    return defs


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
    connectives_classifications = {
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


# Define NAI, MASEN, and NAI-MASEN using pyrsistent structures
NAI = v(
    m(ORTH=m(IN=v("ない", "無い", "ず", "ぬ", "なかっ", "無かっ", "ねえ"))),
    m(ORTH="た", OP="?"),
)

MASEN = s(
    v(m(ORTH="ませ"), m(ORTH="ん")),
    v(m(ORTH="ませ"), m(ORTH="ん"), m(ORTH="でし"), m(ORTH="た")),
)

NAI_MASEN = MASEN.add(NAI)

# Define other structures
ARU = s(
    v(
        m(LEMMA=m(IN=v("ある", "有る")), ORTH=m(NOT_IN=v("あろう"))),
        m(ORTH="た", OP="?"),
    ),
    v(m(NORM="有る"), m(ORTH="ます")),
    v(m(NORM="有る"), m(ORTH="まし"), m(ORTH="た")),
)

ARIMASEN = s(
    v(m(LEMMA=m(IN=v("有る", "ある"))), m(ORTH="ませ"), m(ORTH="ん")),
    v(
        m(LEMMA=m(IN=v("有る", "ある"))),
        m(ORTH="ませ"),
        m(ORTH="ん"),
        m(ORTH="でし"),
        m(ORTH="た"),
    ),
)

DA = s(
    v(
        m(ORTH=m(IN=v("だ", "です", "でし", "だっ")), NORM=m(NOT_IN=v("た"))),
        m(ORTH="た", OP="?"),
    ),
    v(m(ORTH="で"), ARU),
)

DEWA = s(v(m(ORTH="で"), m(ORTH="は")), v(m(ORTH="じゃ")))

VERB_TE = m(ORTH=m(IN=v("て", "で")))

DAROU = s(
    v(m(ORTH=m(IN=v("だろう", "でしょう", "だろ", "たろう")))),
    v(m(ORTH="で"), m(ORTH="あろう")),
)

NOUN_NEGATION = v(DEWA, s(NAI, ARIMASEN))

KOTO = m(NORM=m(IN=v("こと", "事")))


modality_patterns_2 = {
    "だ": {
        "category": ["3認識", "断定"],
        "examples": ["駄目だった", "これは本だ", "この研究は有益だった。"],
        "pattern": DA,
    },
    "のだ": {
        "category": ["4説明", "説明"],
        "examples": ["私のペンなのだ", "彼の成功は努力の結果なのだ。"],
        "pattern": v(m(NORM="の"), DA),
    },
    "という": {
        "category": [],
        "examples": ["発掘されたという"],
        "pattern": v(m(ORTH="と"), m(LEMMA=m(IN=v("言う", "いう")))),
    },
    "だろう": {
        "category": [],
        "examples": ["明日は雨だろう", "対比であろう"],
        "pattern": v(DAROU),
    },
    "となる": {
        "category": [],
        "examples": ["最終的な結果となる"],
        "pattern": v(m(ORTH="と"), m(LEMMA=m(IN=v("成る", "なる")))),
    },
    "とする": {
        "category": [],
        "examples": ["これを基準とする"],
        "pattern": v(m(ORTH="と"), m(LEMMA=m(IN=v("為る", "する")))),
    },
    "ものだ": {
        "category": [],
        "examples": ["避けられないものだ"],
        "pattern": v(m(NORM=m(IN=v("物", "もの"))), DA),
    },
    "ことができる": {
        "category": [],
        "examples": ["解決することができる"],
        "pattern": v(KOTO, m(ORTH="が"), m(NORM="出来る")),
    },
    "と思う": {
        "category": [],
        "examples": ["防げると思う"],
        "pattern": v(m(ORTH="と"), m(NORM="思う", MORPH=m(REGEX="連用形|終止形"))),
    },
    "ことになる": {
        "category": [],
        "examples": ["ことになる"],
        "pattern": v(KOTO, m(ORTH="に"), m(LEMMA=m(IN=v("成る", "なる")))),
    },
    "ことだ": {"category": [], "examples": ["解決することだ"], "pattern": v(KOTO, DA)},
    "ではない": {
        "category": [],
        "examples": ["偶然ではない"],
        "pattern": NOUN_NEGATION,
    },
    "のか": {
        "category": [],
        "examples": ["偶然なのか"],
        "pattern": v(m(NORM="の"), m(ORTH="か")),
    },
    "たい": {
        "category": [],
        "examples": ["検討されたい"],
        "pattern": v(m(ORTH="たい", POS="AUX")),
    },
    "からだ": {
        "category": [],
        "examples": ["成功したからだ"],
        "pattern": v(m(ORTH="から"), DA),
    },
    "ようになる": {
        "category": [],
        "examples": ["できるようになる"],
        "pattern": v(m(ORTH="よう"), m(ORTH="に"), m(LEMMA=m(IN=v("成る", "なる")))),
    },
    "と考えられる": {
        "category": [],
        "examples": ["適切であると考えられる"],
        "pattern": v(m(ORTH="と"), m(NORM="考える"), m(NORM="られる")),
    },
    "ようだ": {
        "category": [],
        "examples": ["適切のようだ"],
        "pattern": v(m(ORTH="よう"), DA),
    },
    "にする": {
        "category": [],
        "examples": ["目標にする"],
        "pattern": v(m(NORM="に"), m(NORM="為る")),
    },
    "なければならない": {
        "category": [],
        "examples": ["努力しなければなりません"],
        "pattern": v(
            m(ORTH="なけれ"), m(ORTH="ば"), m(ORTH=m(IN=v("なら", "なり"))), NAI_MASEN
        ),
    },
    "のだろう": {
        "category": [],
        "examples": ["偶然なのだろう"],
        "pattern": v(m(NORM="の"), DAROU),
    },
    "といえる": {
        "category": [],
        "examples": ["努力の結果と言える"],
        "pattern": v(m(ORTH="と"), m(LEMMA=m(IN=v("言える", "いえる")))),
    },
    "かもしれない": {
        "category": [],
        "examples": ["誤差によるものかもしれない"],
        "pattern": v(m(ORTH="か"), m(ORTH="も"), m(NORM="知れる"), NAI_MASEN),
    },
    "とした": {
        "category": [],
        "examples": ["目標とした"],
        "pattern": v(m(ORTH="と"), m(ORTH="し"), m(ORTH="た")),
    },
    "必要がある": {
        "category": [],
        "examples": ["採掘する必要がある"],
        "pattern": v(m(NORM="必要"), m(ORTH="が"), ARU),
    },
    "のだろうか": {
        "category": [],
        "examples": ["誤差なのだろうか"],
        "pattern": v(m(NORM="の"), DAROU, m(ORTH="か")),
    },
    "わけだ": {
        "category": [],
        "examples": ["成功したわけだ"],
        "pattern": v(m(ORTH=m(IN=v("わけ", "訳"))), DA),
    },
    "ことがある": {
        "category": [],
        "examples": ["誤作動することがある"],
        "pattern": v(KOTO, m(ORTH="が"), ARU),
    },
    "そうだ": {
        "category": [],
        "examples": ["検討するそうだ"],
        "pattern": v(m(MORPH=m(REGEX="^((?!連用形-一般).)*$")), m(ORTH="そう"), DA),
    },
    "ということだ": {
        "category": [],
        "examples": ["低い結果だということだ"],
        "pattern": v(m(ORTH="と"), m(LEMMA=m(IN=v("言う", "いう"))), KOTO, DA),
    },
    "だろうか": {
        "category": ["1表現類型", "疑問"],
        "examples": ["解決するだろうか"],
        "pattern": v(DAROU, m(ORTH="か")),
    },
    "はずだ": {
        "category": [],
        "examples": ["成功するはずだ"],
        "pattern": v(m(NORM="筈"), DA),
    },
    "らしい": {
        "category": [],
        "examples": ["成功するらしい"],
        "pattern": v(m(ORTH="らしい")),
    },
    "ことにする": {
        "category": [],
        "examples": ["再考することにする"],
        "pattern": v(m(NORM="こと"), m(ORTH="に"), m(LEMMA="する")),
    },
    "てよい": {
        "category": [],
        "examples": ["設置してよい"],
        "pattern": v(VERB_TE, m(NORM="良い")),
    },
    "と思われる": {
        "category": [],
        "examples": ["その原因と思われる"],
        "pattern": v(m(ORTH="と"), m(NORM="思う"), m(NORM="れる")),
    },
    "わけではない": {
        "category": [],
        "examples": ["変更できないわけではありません"],
        "pattern": v(m(ORTH=m(IN=v("わけ", "訳"))), NOUN_NEGATION),
    },
    "ことはない": {
        "category": [],
        "examples": ["外れることはない"],
        "pattern": v(KOTO, m(ORTH="は"), s(NAI, v(m(NORM="有る"), MASEN))),
    },
    "ではないか": {
        "category": [],
        "examples": ["その結果ではないか"],
        "pattern": v(NOUN_NEGATION, m(ORTH="か")),
    },
    "なさい": {
        "category": [],
        "examples": ["論じなさい"],
        "pattern": v(m(NORM="為さる")),
    },
    "のに": {
        "category": [],
        "examples": ["解決できたのに"],
        "pattern": v(m(ORTH="の"), m(ORTH="に")),
    },
    "のではない": {
        "category": [],
        "examples": ["最善の方法なのではない"],
        "pattern": v(m(NORM="の"), NOUN_NEGATION),
    },
    "うる": {
        "category": [],
        "examples": ["有効な方法でありうる"],
        "pattern": v(m(POS="VERB"), m(NORM="得る")),
    },
    "べきだ": {
        "category": [],
        "examples": ["新しい方法を開発すべきだ"],
        "pattern": v(m(ORTH="べき"), DA),
    },
    "に違いない": {
        "category": [],
        "examples": ["有効な方法に違いない"],
        "pattern": v(m(ORTH="に"), m(NORM="違い"), s(NAI, v(m(NORM="有る"), MASEN))),
    },
    "てはならない": {
        "category": [],
        "examples": ["損害する方法を開発してはならない"],
        "pattern": v(VERB_TE, m(ORTH="は"), m(NORM="成る"), NAI_MASEN),
    },
    "のではないか": {
        "category": [],
        "examples": ["最善の方法なのではないか"],
        "pattern": v(m(NORM="の"), NOUN_NEGATION, m(ORTH="か")),
    },
    "ものではない": {
        "category": [],
        "examples": ["簡易なものではない"],
        "pattern": v(m(NORM="物"), NOUN_NEGATION),
    },
    "ことだろう": {
        "category": [],
        "examples": ["有効な方法であることだろう"],
        "pattern": v(KOTO, DAROU),
    },
    "とされている": {
        "category": [],
        "examples": ["有効な方法とされている"],
        "pattern": v(m(ORTH="と"), m(ORTH="さ"), m(NORM="れる"), VERB_TE, m(NORM="居る")),
    },
    "のではないだろうか": {
        "category": [],
        "examples": ["最善の方法なのではないだろうか"],
        "pattern": v(m(NORM="の"), NOUN_NEGATION, DAROU, m(ORTH="か")),
    },
    "ものとする": {
        "category": [],
        "examples": ["乙のものとする"],
        "pattern": v(m(NORM="物"), m(ORTH="と"), m(NORM="為る")),
    },
    "ないか": {
        "category": [],
        "examples": ["決定しないか"],
        "pattern": v(NAI, m(ORTH="か")),
    },
    "てほしい": {
        "category": [],
        "examples": ["検討してほしい"],
        "pattern": v(VERB_TE, m(NORM="欲しい")),
    },
    "でもない": {
        "category": [],
        "examples": ["最善の方法でもない"],
        "pattern": v(m(ORTH="で"), m(ORTH="も"), s(NAI, v(m(NORM="有る"), MASEN))),
    },
    "にすぎない": {
        "category": [],
        "examples": ["推測に過ぎない"],
        "pattern": v(m(ORTH="に"), m(NORM="過ぎる"), NAI_MASEN),
    },
    "が考えられる": {
        "category": [],
        "examples": ["方法が考えられる"],
        "pattern": v(m(ORTH="が"), m(NORM="考える"), m(NORM="られる")),
    },
    "まい": {
        "category": [],
        "examples": ["失敗するまい"],
        "pattern": v(m(POS="AUX", ORTH="まい")),
    },
    "といわれている": {
        "category": [],
        "examples": ["有効な方法と言われている"],
        "pattern": v(m(ORTH="と"), m(NORM="言う"), m(NORM="れる"), VERB_TE, m(NORM="居る")),
    },
    "ざるをえない": {
        "category": [],
        "examples": ["開発せざるをえない"],
        "pattern": v(m(ORTH="ざる"), m(ORTH="を"), m(NORM="得る"), NAI_MASEN),
    },
    "ではないだろうか": {
        "category": [],
        "examples": ["最善の方法ではないだろうか"],
        "pattern": v(
            m(ORTH="で"),
            m(ORTH="は"),
            s(m(ORTH="なかろう"), v(NAI, DAROU)),
            m(ORTH="か"),
        ),
    },
    "てしまう": {
        "category": ["考えてしまった"],
        "examples": ["新しい方法を開発してしまう"],
        "pattern": v(VERB_TE, m(NORM="仕舞う")),
    },
    "（よ）う": {
        "category": ["1表現類型", "勧誘"],
        "examples": ["行こう", "新しい方法を開発しよう"],
        "pattern": v(m(POS=m(IN=v("VERB", "AUX")), MORPH=m(REGEX="意志推量形"))),
    },
}

dm_patterns = {"": {"category": [], "examples": [], "pattern": v()}}

if __name__ == "__main__":
    print(modality_patterns)
    print(modality_patterns_2)
    print(connectives_patterns)
    print(connectives_regexes)
    print(connectives_classifications)
