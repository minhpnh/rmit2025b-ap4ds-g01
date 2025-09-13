import re
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from typing import Dict, List
from flask import current_app


class Tokenizer:
    def __init__(self) -> None:
        # Pre-load resources once
        self.token_pattern = re.compile(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?")
        self.lemmatizer = WordNetLemmatizer()
        self.spell = SpellChecker()

        # You’ll probably want to load your stopwords, ignore_words, approved_fixes, etc.
        with open("./data/stopwords_en.txt", "r", encoding="utf-8") as f:
            self.stopwords = {line.strip().lower() for line in f if line.strip()}

        self.approved_fixes = {
            "accomodate": "accommodate",
            "ahd": "and",
            "alittle": "little",
            "back-ordered": "backordered",
            "baby-doll": "babydoll",
            "blk": "black",  # common shorthand → color
            "bralettes": "bralette",  # aligns with your noun lemmatization
            "close-up": "closeup",
            "co-worker": "coworker",
            "curvey": "curvy",
            "cut-outs": "cutouts",
            "dissapointed": "disappointed",
            "fabri": "fabric",
            "fold-over": "foldover",
            "half-way": "halfway",
            "inbetween": "in-between",
            "blue-ish": "bluish",
            "layed": "laid",
            "bac": "back",
            "baily": "bailey",  # common name; suggestion "badly" is wrong
            "make-up": "makeup",
            "orde": "order",
            "oth": "both",
            "peite": "petite",
            "pilcro's": "pilcro",  # brand name, normalize possessive
            "retailerpolgie": "retailergologie",  # looks like misspelling of retailer + brand
            "sotre": "store",
            "stiching": "stitching",
            "sandles": "sandals",
            "wasnt": "wasn't",
            "wayyy": "way",
            "absolutley": "absolutely",
            "amterial": "material",
            "and-go": "and-go",  # but safer to normalize → "on-the-go"? (decide based on context)
            "applique": "applique",  # normalize spelling without accent
            "arm-holes": "armholes",
            "atleast": "at least",  # normalize
            "balloony": "balloon",
            "becuase": "because",
            "blouse-y": "blousey",
            "buyi": "buy",
            "camisol": "camisole",
            "colo": "color",
            "colr": "color",
            "coul": "could",
            "detial": "detail",
            "dresse": "dress",
            "embroiding": "embroidering",
            "eptite": "petite",
            "fabr": "fabric",
            "high-wasted": "high-waisted",
            "hmmm": "hmm",
            "jus": "just",
            "lavendar": "lavender",
            "lenght": "length",
            "long-ish": "longish",
            "long-sleeves": "long-sleeve",
            "long-time": "longtime",
            "lov": "love",
            "middl": "middle",
            "midweight": "mid-weight",
            "muc": "much",
            "nad": "and",
            "non-existent": "nonexistent",
            "non-stop": "nonstop",
            "nto": "to",
            "orderd": "order",
            "ordere": "order",
            "over-priced": "overpriced",
            "perf": "perfect",
            "pettie": "petite",
            "pilcrow": "pilcro",
            "prett": "pretty",
            "push-up": "pushup",
            "recommen": "recommend",
            "roder": "order",
            "sandels": "sandals",
            "seater": "sweater",
            "see-though": "see-through",
            "shee": "she",
            "shir": "shirt",
            "shoul": "should",
            "skintone": "skin-tone",
            "snd": "and",
            "sti": "still",  # or "sit" if that fits context better; choose one
            "str": "star",
            "su": "so",
            "summ": "sum",
            "tee-shirt": "t-shirt",
            "throu": "through",
            "toosh": "too",  # if context is body part (toosh = tush), you might prefer ignore
            "tru": "true",  # or "try" — needs dataset context
            "tule": "tulle",  # fabric type
            "tw": "to",
            "underneat": "underneath",
            "vey": "very",
            "volumous": "voluminous",
            "waaaay": "way",
            "wais": "waist",
            "wayyyy": "way",
            "wearin": "wearing",
            "whic": "which",
            "wri": "write",
            "xtra": "extra",
            "absolutly": "absolutely",
            "accordian": "accordion",
            "ador": "adore",
            "agin": "again",
            "ahhh": "ah",
            "ahs": "as",
            "aize": "size",
            "apprx": "approx",
            "attache": "attaché",  # can also simplify to "attached"
            "beautifu": "beautiful",
            "becaus": "because",
            "bein": "being",
            "bette": "better",
            "blu": "blue",
            "bottons": "buttons",
            "boyleg": "boy-leg",  # normalize to clothing style
            "breton": "breton",  # it's actually a valid fashion term, but if you want correction → keep as-is
            "bust-line": "bustline",
            "buttoned-up": "button-up",
            "cehst": "chest",
            "chrolox": "clorox",  # likely brand typo
            "cld": "cold",  # though could also be short for "could"; context-dependent
            "clinged": "clung",  # better fix than "cringed"
            "col": "color",  # often abbreviation
            "comfotable": "comfortable",
            "comfrotable": "comfortable",
            "completel": "completely",
            "compliements": "compliments",
            "cou": "you",
            "craftmanship": "craftsmanship",
            "criss-crosses": "crisscrosses",
            "crossbody": "cross-body",  # normalize
            "crosswrap": "cross-wrap",  # normalize
            "cu": "cut",
            "cut-off": "cutoff",
            "dangly": "dangle",
            "decolletage": "décolletage",
            "definetly": "definitely",
            "defini": "define",
            "definitly": "definitely",
            "denimy": "denim",
            "did't": "didn't",
            "differen": "different",
            "disappoin": "disappoint",
            "do-able": "doable",
            "drape-y": "drapey",
            "do-over": "do-over",  # keep valid phrase, don't change to hoover
            "eit": "it",
            "ejans": "jeans",
            "eprson": "person",
            "eptites": "petites",
            "especia": "especial",
            "everthing": "everything",
            "every-day": "everyday",
            "excelle": "excellent",  # better normalization than "excelled"
            "exmaple": "example",
            "flatte": "flatter",
            "flattrering": "flattering",
            "flatttering": "flattering",
            "forwar": "forward",
            "ful": "full",
            "fuschia": "fuchsia",
            "gauzey": "gauzy",
            "gawdy": "gaudy",
            "gianormous": "ginormous",
            "giv": "give",
            "grea": "great",
            "hand-washed": "hand-wash",
            "high-waist": "high-waisted",
            "higher-waisted": "high-waisted",
            "highwaisted": "high-waisted",
            "hink": "think",
            "howeve": "however",
            "huuuge": "huge",
            "hwoever": "whoever",
            "hypen": "hyphen",
            "ike": "like",
            "isn": "isn't",
            "henleys": "henley",
            "kansa": "kansas",
            "kno": "know",
            "kne": "knee",
            "large-ish": "largish",
            "lat": "at",
            "leary": "leery",
            "legnth": "length",
            "liekd": "liked",
            "lightwei": "lightweight",
            "litt": "little",
            "littl": "little",
            "llb": "llbean",  # common brand in clothing reviews (L.L.Bean)
            "llonger": "longer",
            "lnever": "never",
            "loke": "like",
            "lon": "long",  # usually truncation
            "loompa": "oompa",  # often from “Oompa Loompa”
            "looove": "love",
            "los": "loss",  # better than “lot”
            "lotta": "lot of",
            "low-waisted": "low-waisted",
            "lucious": "luscious",
            "luv": "love",
            "lvoe": "love",
            "lwait": "wait",
            "mateiral": "material",
            "materia": "material",
            "materiel": "material",
            "mauve-ish": "mauvish",
            "mentioend": "mentioned",
            "metalic": "metallic",
            "mid-riff": "midriff",
            "midrise": "mid-rise",
            "mary": "mary",  # keep as name, but if context shows typo of “many” → map to "many"
            "marroon": "maroon",
            "marilyn": "marilyn",  # valid name (keep unless you want to normalize to “marlin”)
            "martie": "martie",  # name, but if typo → "martin"
            "mexican": "mexican",  # nationality, keep as-is
            "multi-colored": "multicolored",
            "neede": "need",
            "nicel": "nice",
            "non-issue": "nonissue",
            "non-traditional": "nontraditional",
            "noticable": "noticeable",
            "offwhite": "off-white",  # if you prefer to normalize; otherwise keep in ignore_words
            "one-size": "one-size",  # keep as sizing token; do NOT change to "oversize"
            "onsie": "onesie",
            "othe": "the",
            "ou": "you",
            "ov": "of",
            "over-dress": "overdress",
            "over-lay": "overlay",
            "pai": "pay",
            "peice": "piece",
            "perfe": "perfect",
            "perhap": "perhaps",
            "petities": "petites",
            "petitte": "petite",
            "pilcos": "pilco",
            "pictur": "picture",
            "lyocel": "lyocell",
            "lucious": "luscious",  # if not already added earlier
            "plasticy": "plasticky",  # more natural than "plastic" for texture
            "pool-side": "poolside",
            "popback": "pop-back",
            "post-partum": "postpartum",
            "potatoe": "potato",
            "prefectly": "perfectly",
            "re-ordering": "reordering",
            "re-stock": "restock",
            "reall": "really",
            "recomend": "recommend",
            "rediculously": "ridiculously",
            "referance": "reference",
            "regulat": "regular",
            "retuned": "returned",
            "righ": "right",
            "risqu": "risqué",
            "seea": "see",
            "shap": "shape",
            "shld": "should",
            "petities": "petites",
            "petitte": "petite",
            "pictur": "picture",
            "pilcos": "pilco",
            "purcha": "purchase",
            "purc": "purchase",
            "quali": "quality",
            # keep earlier decisions consistent:
            "over-dress": "overdress",
            "over-lay": "overlay",
            "recomme": "recommend",
            "shou": "show",
            "show-stopper": "showstopper",
            "sie": "she",
            "silouette": "silhouette",
            "size-i": "size",
            "sizi": "size",
            "sleevless": "sleeveless",
            "slighly": "slightly",
            "slighty": "slightly",
            "smaill": "small",
            "sofisticated": "sophisticated",
            "stretc": "stretch",
            "styl": "style",
            "summe": "summer",
            "sweater-coat": "sweatercoat",  # if you prefer preserving hyphen, move to ignore_words instead
            "t'shirt": "t-shirt",
            "taupey": "taupe",
            "taylored": "tailored",
            "tey": "they",
            "th": "the",
            "thes": "the",
            "thier": "their",
            "thin-ish": "thinnish",
            "thre": "the",
            "throw-away": "throwaway",
            "thsi": "this",
            "tight-fitting": "tightfitting",
            "tme": "the",
            "toget": "get",
            "togethers": "together",
            # keep “too-small” as a valid descriptor (see ignore below)
            "tyhlo": "typo",
            "un-button": "unbutton",
            "unflatering": "unflattering",
            "unfortunatly": "unfortunately",
            "uppe": "upper",
            "undernea": "underneath",  # align with earlier “underneat” → “underneath”
            "waht": "what",
            "warm-up": "warmup",
            "week-end": "weekend",
            "wel": "we",
            "wiast": "waist",
            "xlarge": "large",
            "xxsmall": "x-small",
            "youre": "your",
            "unfort": "unfortunately",  # common truncation in reviews
        }
        self.ignore_words = [
            "ada",  # could be name/acronym/brand
            "alaska",
            "as-is",
            "atlanta",
            "ankle-length",
            "american",
            "basketweave",
            "bermuda",
            "birkenstocks",
            "blue-green",
            "boatneck",
            "body-hugging",
            "body-skimming",
            "boston",
            "charlie's",
            "cowlneck",
            "criss",  # often from "criss-cross"
            "d's",  # bra sizing context possible
            "da",  # dialect/name; avoid auto-fix
            "day-to",  # likely part of "day-to-day"
            "dddd",  # sizing shorthand (e.g., DDD)
            "drop-waist",
            "flip-flop",
            "gauze-like",
            "handwash",
            "high-end",
            "honolulu",
            "hoxton",  # style/brand/cut
            "in-between",  # already normalized target exists
            "less-than",
            "april",  # month
            "levi's",  # brand
            "lola",  # name
            "lyocell",  # fabric
            "machine-washable",
            "marrakech",  # place
            "medium-weight",
            "miami",  # place
            "mid-knee",
            "mih",  # brand (MIH Jeans)
            "mockneck",  # clothing style
            "neira",  # name
            "no-brainer",
            "no-go",  # valid slang
            "off-center",
            "oop",  # could be slang ("out of print", "oops")
            "post-pregnancy",
            "rosie",  # name
            "s-m",  # size
            "sacklike",
            "se",  # abbreviation (Southeast, Spanish "se")
            "shapewear",
            "sl",  # abbreviation
            "small-medium",  # size
            "socal",  # region
            "super-flattering",
            "tentlike",
            "tie-dye",
            "tie-neck",
            "uk",  # country
            "und",  # German word for "and"
            "underslip",
            "v-cut",
            "v-shape",
            "vee",  # neckline type
            "verdugo",  # denim brand
            "pd",  # abbreviation
            "well-done",
            "well-endowed",
            "well-fitting",  # correction to "ill-fitting" is wrong
            "xmas",  # common short form
            "abo",  # could be abbreviation
            "above-the",
            "activewear",
            "age-appropriate",
            "air-dried",
            "alexandria",  # place
            "amadi",  # brand/name
            "amalfi",  # place
            "anth",  # shorthand for Anthropologie
            "app",  # app (application)
            "aren",  # could be "aren’t", leave as context check
            "as-pictured",
            "asia",
            "athleisure",
            "bam",  # slang
            "bbq",  # abbreviation
            "bday",  # short for birthday
            "beca",  # name
            "becau",  # likely cut-off typing, but too ambiguous
            "betty",  # name
            "bl",  # abbreviation (black, blouse, etc.)
            "black-and",  # part of compound "black-and-white"
            "bling",  # slang
            "body-type",
            "boho-chic",
            "bootcut",
            "bot",  # slang/chat context
            "british",
            "business-casual",
            "cafe",
            "cami's",  # possessive
            "cardio",
            "carissima",  # Italian word/name
            "chanel",  # brand
            "clo",  # could be CLO unit/abbr; ambiguous
            "color-wise",
            "colorblock",
            "crewneck",
            "cupro",  # fabric
            "dallas",
            "dara",  # name
            "defin",  # likely truncated “definitely”; avoid wrong fix
            "delicates",  # valid laundry term
            "diego",
            "dn",  # abbreviation
            "double-layered",
            "double-sided",
            "downton",  # could be “Downton”
            "dre",  # truncated; ambiguous
            "dryel",  # brand (home dry-cleaning)
            "dy",  # ambiguous
            "easy-to",
            "evanthe",  # name/brand
            "ever-so",
            "faux-fur",
            "feb",  # month abbr
            "fiance",  # valid word (don’t change to finance)
            "fr",
            "ge",
            "gr",
            "gre",  # ambiguous short tokens
            "grecian",
            "hh",
            "hollywood",
            "ia",
            "ibs",  # abbreviations
            "irish",
            "january",
            "kate",
            "ke",
            "keds",
            "kn",
            "maternity-like",
            "mid-hip",
            "midwest",
            "missoni",  # brand
            "mona",  # name
            "monday",  # weekday
            "mumu",  # garment style
            "no-fuss",
            "non-stretchy",
            "nouveau",
            "nye",  # New Year's Eve
            "onesie",
            "op",  # abbreviation
            "orleans",  # place
            "over-shirt",  # fashion term; if you prefer, later normalize -> "overshirt"
            "paquerette",  # name/line
            "pepto",  # brand
            "polka-dots",
            "prima",  # valid word
            "pur",  # ambiguous abbrev/brand
            "raf",  # acronym
            "ranna",  # designer (Ranna Gill)
            "rec",  # abbreviation (recommend/receipt/record)
            "red-orange",
            "sc",  # abbreviation (e.g., South Carolina)
            "seasonless",
            "self-conscious",
            "sewn-in",
            "shearling",  # valid fabric
            "short-sleeved",
            "show-through",
            "size-wise",
            "skin-toned",
            "skorts",  # valid clothing type
            "sle",  # ambiguous abbreviation
            "small-framed",
            "super-sale",
            "swea",  # truncation, risky
            "t-back",  # lingerie style
            "top-heavy",  # valid descriptor
            "tucked-in",  # style term
            "umph",  # interjection
            "uncuff",  # style term
            "ur",  # abbreviation for “your”
            "velcro",  # brand
            "washed-out",
            "well-worn",
            "whi",  # truncation, ambiguous
            "win-win",
            "wishlist",
            "woolite",  # brand
            "two-tone",  # valid style term
            "x-large",
            "x-s",
            "xl's",
            "xxxl",  # sizing
            "a-cup",  # bra size
            "a-flutter",  # valid expression
            "a-frame",  # style/structure
            "a-kind",  # phrase
            "a-symmetric",  # variant of asymmetric
            "abby",  # name
            "abou",  # truncated "about", but risky
            "add-on",  # common term
            "ag's",  # abbreviation
            "all-around",
            "all-in",
            "all-over",
            "allison",  # name
            "amd",  # typo for "and", but also common abbrev (Advanced Micro Devices)
            "angeles",  # Los Angeles
            "ann's",  # possessive
            "apple-shaped",  # body description
            "arielle",  # name
            "arty-looking",  # valid descriptor
            "ashley",  # name
            "atl",  # abbreviation for Atlanta
            "az",  # abbreviation for Arizona
            "b-c",
            "ba",  # ambiguous abbrevs
            "back-up",  # valid expression
            "bea",  # name
            "bec",  # abbreviation
            "bell-sleeve",  # clothing style
            "bf",
            "bff",  # slang
            "blue-grey",  # color
            "bluishgreen",  # color
            "boat-neck",  # clothing style
            "bodytype",  # descriptor
            "bomber-style",  # clothing style
            "bottom-heavy",  # descriptor
            "bra-straps",  # clothing
            "broad-shouldered",
            "brooklyn",  # place
            "btu",  # abbreviation (British Thermal Unit)
            "bubble-like",
            "button-front",
            "c-cup",
            "c-d",  # bra sizes
            "cali",  # short for California
            "canvas-y",  # fashion descriptor
            "cardis",  # shorthand for cardigans
            "casu",  # shorthand for casual
            "cc",  # abbreviation
            "charleston",  # place
            "children's",
            "coh's",  # could be brand/possessive
            "color-blocking",
            "colorblocked",
            "colorwise",
            "coloured",  # fashion terms / accepted spellings
            "corodorys",  # might be variant misspelling of corduroys; if frequent, consider adding fix
            "denver",  # place
            "dept",  # abbreviation for department
            "desi",  # proper noun / cultural term
            "dolman-style",  # clothing type
            "double-lined",
            "double-v",
            "druzy",  # gem term
            "dry-clean",
            "dry-cleaning",  # valid clothing care instructions
            "dryclean",  # garment care term
            "durham",  # place name
            "earth-tone",  # valid descriptor
            "easy-breezy",  # valid phrase
            "eira",  # name
            "eloise",  # name
            "else's",  # valid possessive
            "emb",  # abbreviation
            "endora",  # name
            "erin",  # name
            "errand-running",  # phrase
            "eu",  # abbrev/region
            "everleigh",  # name/brand
            "ewww",  # expression
            "extra-large",  # sizing
            "fairisle",  # knitting style
            "fall-winter",  # fashion season
            "farrah",  # name
            "favourite",  # valid British spelling
            "february",  # month
            "felty",  # texture term
            "femine",  # ambiguous, could be mistaken spelling but not common enough
            "feminie",  # same, could map → feminine, but might distort rare text
            "fetherston",  # designer (Erin Fetherston)
            "ff",  # abbreviation
            "fianc",  # possibly truncation of fiancé; safer to ignore
            "figure-hugging",  # style term
            "filipino",  # nationality, valid
            "fit-wise",  # valid descriptor
            "five-year",  # phrase
            "fla",  # abbreviation/slang
            "fleetwood",  # name
            "flesh-colored",  # valid descriptor
            "flirtiness",  # valid word
            "flo",  # name/slang
            "flowier",  # valid comparative
            "foldover",  # style term
            "form-fitted",  # valid fashion descriptor
            "france",  # place
            "full-time",  # phrase
            "fully-lined",  # descriptor
            "g's",  # slang
            "ga",  # abbreviation/state
            "expe",  # ambiguous truncation (expect/expense/experience)
            "georgette",  # fabric
            "go-anywhere",  # valid phrase
            "greek",  # valid nationality/descriptor
            "hahaha",  # expression
            "hand-knit",  # descriptor
            "haute",  # fashion term
            "hei-hei",  # brand (Anthropologie)
            "high-heeled",  # descriptor
            "hipline",  # garment measurement term
            "hol",  # ambiguous truncation (holiday/hold)
            "hook-and",  # part of "hook-and-eye/loop"
            "housedress",  # garment
            "hr",  # abbreviation (hour/human resources)
            "inbox",  # valid word
            "iphone",  # product
            "isabella",  # name
            "italian",  # nationality
            "itty",  # valid word
            "jackie",
            "japanese",
            "jeera",
            "joan",
            "joe's",  # names/valid words
            "jammie",  # colloquial (pajamas)
            "ju",
            "juuuust",  # stylized emphasis
            "kedia",  # uncertain; leave as-is
            "kentucky",
            "kim",  # proper nouns
            "l-xl",  # size token
            "lace-like",  # descriptor
            "levi",  # brand
            "jewel-tone",  # valid fashion term
            "knit-like",  # valid descriptor
            "light-colored",
            "light-to",
            "little-girl",
            "looooooove",  # stylized
            "loose-fit",  # descriptor
            "louisiana",  # place
            "maternity-esque",  # style descriptor
            "mcguire",  # brand
            "meda",  # possible name
            "mediterranean",  # region
            "medium-to",  # descriptor
            "mid-length",
            "mid-september",
            "mid-shin",
            "midi-length",
            "minneapolis",  # place
            "mismarked",  # valid term
            "mixed-media",  # descriptor
            "mmmmm",  # stylized
            "mn",  # abbreviation
            "mock-neck",  # descriptor
            "lin",  # could be name (Lin)
            "london",  # place
            "lycra",  # fabric
            "might've",  # valid contraction
            "min",  # ambiguous (minute/minimum)
            "mini-dress",  # style term
            "mom-bod",  # descriptor
            "monica",  # name
            "moo-moo",  # variant of muumuu; keep as written
            "muffin-top",  # descriptor
            "mui",  # ambiguous (brand/abbr)
            "must-buy",  # phrase
            "napa",  # place
            "newport",  # place/brand
            "non-bulky",
            "non-flattering",
            "non-maternity",  # descriptive hyphen forms
            "nordstrom",  # store/brand
            "not-too",  # phrase (often part of compound)
            "november",  # month
            "nude-colored",  # descriptor
            "ny",  # New York abbrev
            "off-shoulder",  # style term
            # If you chose to normalize offwhite above, remove it from ignore_words; else keep it here:
            # "offwhite",
            "one-of",  # phrase
            "ons",  # ambiguous abbrev
            "oompa",  # part of “Oompa Loompa”
            "open-weave",  # descriptor
            "orange-y",
            "orangish",  # color descriptors
            "otk",  # over-the-knee (boots)
            "overnighted",  # valid past tense verb
            "pajama-like",  # descriptor
            "pants-they",  # punctuation artifact; safest to leave
            "petite-sized",  # descriptor
            "photoshoot",
            "photoshopped",
            "pilly",  # real adjective (fabric pilling)
            "pintucks",
            "pippa",  # name
            "pkg",  # abbreviation (package)
            "polka-dot",
            "polkadots",
            "poncho-like",
            "poncho-type",
            "poofiness",  # valid word in fashion context
            "pop-back",  # product/style term in some listings
            "pre-baby",
            "pre-ordered",
            "pre-washed",
            "pre-wedding",  # valid “pre-” forms
            "preggers",
            "preggo",
            "prego",  # slang; keep if you want to preserve tone
            "pricepoint",  # common retail term
            "proba",
            "produ",  # ambiguous truncations
            "provence",  # place
            "recd",  # “rec’d” / received abbrev
            "regular-length",
            "relaxed-fit",
            "reno",  # place
            "retu",  # ambiguous truncation
            "ri",
            "rica",  # could be RI (state)/Costa Rica
            "rockefeller",  # proper noun
            "roll-up",  # style term
            "rona",  # name
            "rosey",  # earlier you mapped to “rose”; if you prefer adjective, use “rosy” in approved_fixes
            "ru",  # if earlier mapped to "run", remove from ignore to avoid conflict
            "sale-on",  # phrase
            "scotty",  # name
            "semi-fitted",
            "sf",  # abbreviation
            "shearling-lined",
            "shooties",  # legit shoe/bootie style
            "side-zipper",  # style/feature term
            "sinclair",  # proper noun
            "sister-in",  # compound (often part of “sister-in-law”)
            "siyu",  # name/brand
            "sk",  # abbreviation
            "skir",  # ambiguous truncation (could be “skirt”)
            "slee",  # ambiguous truncation
            "sli",
            "slig",  # ambiguous truncations
            "slubby",  # legitimate fabric texture
            "small-ish",  # accepted hyphenated form
            "soft-looking",
            "sooooooo",  # stylized emphasis
            "souers",  # ambiguous (name/typo); safer to keep
            "special-occasion",
            "spetite",  # if you intended to normalize → move to approved_fixes = "petite"
            "spokane",  # place
            "square-apple",  # body-shape descriptor
            "stand-out",  # descriptor (earlier kept)
            "starbucks",  # brand
            "static-y",  # keep as written for consistency (you can normalize → “staticky” if desired)
            "stra",  # ambiguous truncation (star/strap/straight)
            "straight-leg",
            "straight-legged",
            "super-hot",
            "super-skinny",
            "super-tiny",
            "sur",
            "sw",  # abbreviations
            "tailor-fitted",
            "tennies",  # slang for sneakers
            "tenty",  # legit adjective meaning “tent-like”
            "terrycloth-like",
            "tex",  # abbreviation
            "that'd",
            "that'll",  # valid contractions (don’t change to “that’s”)
            "ther",  # ambiguous (the/there/their) — safer to leave
            "size-small",  # size descriptor
            "too-small",  # valid descriptor
            "torsoed",  # fashion/silhouette term; suggestion “tossed” is wrong
            "tuesday",  # weekday
            "tunic-like",  # descriptor
            "tx",  # abbreviation (Texas, transaction, etc.)
            "unco",  # ambiguous truncation
            "uncuffed",  # style term
            "under-slip",  # garment term; suggestion is wrong
            "unseamed",  # valid adjective
            "unstitched",  # valid adjective
            "upside-down",  # descriptor
            "upsize",
            "upsizing",  # valid retail terms
            "usu",  # abbreviation for “usually”
            "v's",  # pluralized letter/size notation
            "v-neckline",  # descriptor
            "versat",  # ambiguous truncation (could be versatile/versatility)
            "vinyasa",  # yoga style
            "wamp",  # ambiguous (name/slang)
            "washability",  # descriptor
            "wedding-ish",  # descriptor
            "well-cut",  # descriptor
            "workwear",  # valid term
            "worn-in",  # descriptor
            "wrinkle-prone",  # descriptor
            "xs's",  # size/plural form
            "xxl",  # size
            "zag",  # valid word (as in zig-zag)
            "tra",  # ambiguous truncation (try/track/trapeze) — safer to keep
            "vintage-looking",  # valid descriptor
            "wid",  # ambiguous (with/would) — you previously kept this; stay consistent
        ]

    def _apply_typo_fixes(self, tokens: List[str]) -> List[str]:  # pyright: ignore typing
        fixed = []
        for t in tokens:
            if t in self.ignore_words:
                fixed.append(t)
            elif t in self.approved_fixes:
                fixed.append(self.approved_fixes[t])
            elif t in self.spell.unknown([t]):
                suggestion = self.spell.correction(t)
                fixed.append(suggestion if suggestion else t)
            else:
                fixed.append(t)
        return fixed

    def process_review(self, review: Dict) -> Dict:
        """
        Given a review dict with at least {"Review Text": "..."},
        return the processed tokens and processed text.
        """
        raw_text = str(review.get("title", "")) + " " + str(review.get("body", ""))

        # Tokenize
        tokens = self.token_pattern.findall(raw_text)

        # Lowercase
        tokens = [t.lower() for t in tokens]

        # Remove short tokens
        tokens = [t for t in tokens if len(t) >= 2]

        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        # Fix typos
        tokens = self._apply_typo_fixes(tokens)

        return {
            "original": raw_text,
            "tokens": tokens,
            "processed_text": " ".join(tokens),
        }
