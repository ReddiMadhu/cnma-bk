"""
build_references.py — One-time setup script.

1. Reads air_static_data.json to generate air_occ_codes.json and air_const_codes.json.
2. Uses existing mapping files (atc_to_air_occ_map.json, rms_to_air_const_map.json)
   to build the complete TF-IDF vocab for RMS codes.
3. Creates abbreviations, iso3166_states, iso4217_currency JSON files.
4. Builds and saves four TF-IDF vectorizer pkl files.

Run from inside the directory:
  python build_references.py
"""
import json
import pathlib
import pickle
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).parent
REF = BASE / "reference"
TFIDF = BASE / "tfidf_cache"
JSON_PATH = REF / "air_static_data.json"

REF.mkdir(exist_ok=True)
TFIDF.mkdir(exist_ok=True)

# ── Check dependencies ─────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as e:
    print(f"Missing dependency: {e}\nRun: pip install scikit-learn")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Read AIR static data from extracted JSON
# ─────────────────────────────────────────────────────────────────────────────

print(f"Reading: {JSON_PATH}")
if not JSON_PATH.exists():
    print(f"ERROR: JSON file not found at {JSON_PATH}. Run extract_air_static_to_json.py first.")
    sys.exit(1)

air_data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

occ_data = air_data.get("occupancyData", [])
const_data = air_data.get("constructionData", [])
biz_map_data = air_data.get("businessTypeOccupMapping", [])

# Build keyword enrichment from businessTypeOccupMapping
biz_keywords = {}
for row in biz_map_data:
    key = str(row.get("key") or "").strip().lower()
    code = str(row.get("value") or "").strip()
    if key and code:
        biz_keywords.setdefault(code, []).append(key)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Generate air_occ_codes.json
# ─────────────────────────────────────────────────────────────────────────────

# Supplemental keyword lists per code (hand-crafted from domain knowledge)
OCC_SUPPLEMENTAL_KEYWORDS = {
    "301": ["general residential", "residential", "dwelling unspecified"],
    "302": ["single family", "single-family", "house", "home", "detached", "sfr", "sfh",
            "residential dwelling", "single family home", "detached house"],
    "303": ["multi family", "multifamily", "duplex", "triplex", "fourplex",
            "multi-family dwelling", "4plex", "multiple family"],
    "304": ["hotel", "motel", "inn", "lodge", "temporary lodging", "accommodation",
            "bed and breakfast", "hostel", "guest house"],
    "305": ["dormitory", "barracks", "group housing", "nursing home", "assisted living",
            "group institutional", "senior living", "care home"],
    "306": ["apartment", "condo", "condominium", "flat", "unit", "apartments condominiums"],
    "307": ["terraced", "townhouse", "rowhouse", "row house", "attached home", "terrace"],
    "311": ["commercial", "business", "general commercial", "mixed use", "office building",
            "commercial building"],
    "312": ["retail", "store", "shop", "mall", "shopping center", "boutique", "showroom"],
    "313": ["wholesale", "distribution", "wholesaler", "distributor", "wholesale trade"],
    "314": ["salon", "repair", "laundry", "dry cleaning", "personal service", "barber",
            "cleaners", "service shop"],
    "315": ["office", "professional", "technical", "business services", "law firm",
            "accounting", "consulting", "professional services", "tech office"],
    "316": ["hospital", "medical", "clinic", "health care", "healthcare", "pharmacy",
            "medical center", "surgery center", "outpatient"],
    "317": ["entertainment", "recreation", "gym", "fitness", "cinema", "theater",
            "swimming pool", "recreation club", "sports facility", "aquatic"],
    "318": ["parking", "garage", "car park", "parking structure", "parking lot"],
    "319": ["golf", "golf course", "country club"],
    "321": ["industrial", "factory", "manufacturing", "plant", "general industrial"],
    "322": ["heavy fabrication", "heavy industrial", "heavy manufacturing", "steel mill",
            "foundry", "smelter", "forge"],
    "323": ["light fabrication", "light manufacturing", "assembly plant", "light industrial"],
    "324": ["food processing", "beverage", "drug processing", "pharmaceutical", "food plant",
            "food facility", "cannery", "brewery", "winery"],
    "325": ["chemical", "chemical plant", "chemical processing", "refinery",
            "chemical facility", "petrochemical"],
    "326": ["metal", "mining processing", "minerals processing", "ore processing"],
    "327": ["high tech", "technology", "semiconductor", "data center", "tech campus",
            "server farm", "lab", "r&d"],
    "328": ["construction", "builder", "contractor", "construction yard", "building supply"],
    "329": ["petroleum", "oil", "oil refinery", "fuel terminal", "petroleum storage"],
    "330": ["mine", "mining", "quarry", "excavation", "mineral extraction"],
    "331": ["restaurant", "cafe", "diner", "food service", "bar", "pub", "eatery",
            "fast food", "cafeteria", "food court"],
    "335": ["gas station", "gasoline station", "petrol station", "fuel station",
            "service station", "filling station"],
    "336": ["auto repair", "car repair", "mechanic", "automotive service", "body shop",
            "auto shop"],
    "341": ["religion", "nonprofit", "non-profit", "charity", "religious organization",
            "foundation", "ngо"],
    "342": ["church", "temple", "mosque", "synagogue", "place of worship", "chapel",
            "cathedral", "religious building"],
    "343": ["government", "general services", "municipal", "city hall", "public building"],
    "344": ["emergency services", "fire station", "police station", "ambulance",
            "first responder", "dispatch center"],
    "345": ["university", "college", "technical school", "campus", "higher education"],
    "346": ["school", "elementary school", "high school", "primary school",
            "secondary school", "k-12", "middle school"],
    "351": ["highway", "road", "transportation hub", "transit station", "bus terminal"],
    "352": ["railroad", "railway", "train station", "rail terminal", "freight yard"],
    "353": ["airport", "air transportation", "aviation facility", "terminal"],
    "354": ["port", "dock", "waterway", "ship terminal", "marina", "harbor"],
    "355": ["hangar", "airplane hangar", "aircraft hangar", "aviation hangar"],
    "356": ["aircraft ramp", "boarding gate", "aircraft at gate"],
    "361": ["electric", "power plant", "electrical utility", "substation", "power station"],
    "362": ["water utility", "water treatment", "water plant", "pumping station"],
    "363": ["sewer", "waste disposal", "wastewater", "sewage treatment"],
    "364": ["natural gas", "gas distribution", "gas utility", "pipeline facility"],
    "365": ["telephone", "telecom", "communication utility", "telephone exchange"],
    "366": ["warehouse", "storage", "distribution center", "logistics", "storage facility"],
    "371": ["communication", "media", "broadcasting", "radio tower", "tv station"],
    "372": ["flood control", "dam", "levee", "flood barrier"],
    "373": ["agriculture", "farm", "farming", "crop", "agricultural building",
            "barn", "grain elevator"],
    "374": ["greenhouse", "nursery", "plant growing", "hothouse"],
    "375": ["forestry", "forest", "timber", "logging", "sawmill"],
}

air_occ_codes = {}
for row in occ_data:
    code = str(row.get("code") or "").strip()
    description = str(row.get("description") or "").strip()
    if not code or not description:
        continue
    keywords = list(OCC_SUPPLEMENTAL_KEYWORDS.get(code, []))
    keywords.extend(biz_keywords.get(code, []))
    keywords.append(description.lower())
    seen_kw = set()
    unique_kw = []
    for kw in keywords:
        if kw not in seen_kw:
            seen_kw.add(kw)
            unique_kw.append(kw)
    air_occ_codes[code] = {"description": description, "keywords": unique_kw}

output_path = REF / "air_occ_codes.json"
output_path.write_text(json.dumps(air_occ_codes, indent=2), encoding="utf-8")
print(f"OK: Wrote {len(air_occ_codes)} AIR occupancy codes -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Generate air_const_codes.json
# ─────────────────────────────────────────────────────────────────────────────

CONST_SUPPLEMENTAL_KEYWORDS = {
    "100": ["unknown", "unspecified", "undefined", "unknown construction"],
    "101": ["wood frame", "modern wood", "lumber frame", "stick frame", "light frame",
            "wf", "wood frame modern", "platform frame modern"],
    "102": ["light wood frame", "light wood", "platform frame", "lwf"],
    "103": ["masonry veneer", "brick veneer", "stone veneer", "veneer construction"],
    "104": ["heavy timber", "post and beam", "mill construction", "heavy wood",
            "post beam", "timber frame"],
    "107": ["lightweight cladding", "metal cladding", "light metal building",
            "pre-engineered metal"],
    "108": ["hale construction", "hale"],
    "111": ["masonry", "brick", "block", "concrete block", "brick masonry",
            "cmu", "concrete masonry unit"],
    "112": ["adobe", "earth brick", "mud brick", "earthen"],
    "113": ["rubble stone", "stone masonry", "fieldstone", "random rubble"],
    "114": ["urm", "unreinforced masonry", "unreinforced masonry bearing wall",
            "unconfined masonry", "plain masonry bearing wall"],
    "115": ["unreinforced masonry bearing frame", "urm frame", "urm with frame"],
    "116": ["reinforced masonry", "rm", "reinforced brick", "reinforced block",
            "reinforced concrete masonry"],
    "117": ["reinforced masonry shear wall mrf", "rm shear wall with mrf",
            "reinforced masonry moment frame"],
    "118": ["reinforced masonry shear wall", "rm shear wall", "rm without mrf"],
    "119": ["joisted masonry", "ordinary construction", "joist masonry",
            "masonry walls wood joists"],
    "120": ["confined masonry", "cm", "confined brick"],
    "121": ["cavity brick", "double brick", "cavity double brick", "hollow brick"],
    "131": ["reinforced concrete", "rc", "concrete frame", "rcc",
            "reinforced concrete frame", "concrete construction"],
    "132": ["rc shear wall mrf", "concrete shear wall mrf", "rc sw mrf"],
    "133": ["rc shear wall", "concrete shear wall", "rc sw", "shear wall concrete"],
    "134": ["rc mrf ductile", "ductile concrete frame", "ductile rc frame",
            "special moment frame concrete"],
    "135": ["rc mrf non-ductile", "non-ductile concrete frame", "non-ductile rc"],
    "136": ["tilt up", "tilt-up", "precast tilt", "tilt wall concrete"],
    "137": ["precast concrete", "pre-cast", "precast", "pc", "precast frame"],
    "138": ["precast concrete shear wall", "pc shear wall"],
    "139": ["rc moment frame", "concrete moment frame", "rc mrf", "ordinary rc mrf"],
    "140": ["rc frame urm", "concrete frame masonry", "rc with urm infill"],
    "141": ["rc frame wood frame", "mixed rc wood", "rc with wood addition"],
    "151": ["steel", "structural steel", "steel construction", "steel frame",
            "steel building"],
    "152": ["light metal", "metal frame", "light steel frame", "light steel",
            "metal building", "pre-engineered steel"],
    "153": ["braced steel", "steel bracing", "eccentrically braced frame",
            "special braced frame", "steel cbf"],
    "154": ["steel mrf perimeter", "perimeter moment frame", "steel smrf perimeter"],
    "155": ["steel mrf distributed", "distributed moment frame", "steel smrf"],
    "156": ["steel moment frame", "steel mrf", "smrf", "steel smf"],
    "157": ["steel urm", "steel frame masonry", "steel with urm infill"],
    "158": ["steel concrete shear wall", "dual system steel concrete"],
    "159": ["src", "steel reinforced concrete", "composite construction",
            "composite steel concrete"],
    "160": ["steel long span", "long span steel", "open web joist steel"],
    "181": ["long span", "open web joist", "warehouse joist"],
    "182": ["semi wind resistive", "semi-resistive", "semi wind resistant"],
    "183": ["wind resistive", "wind resistant", "hurricane resistant"],
    "191": ["mobile home", "manufactured home", "trailer", "mobile home unknown"],
    "192": ["mobile home no tie down", "unanchored mobile home"],
    "193": ["mobile home partial tie down", "semi-anchored mobile home"],
    "194": ["mobile home full tie down", "anchored mobile home", "fully anchored mobile home"],
}

air_const_codes = {}
for row in const_data:
    code = str(row.get("code") or "").strip()
    description = str(row.get("description") or "").strip()
    if not code or not description:
        continue
    keywords = list(CONST_SUPPLEMENTAL_KEYWORDS.get(code, []))
    keywords.append(description.lower())
    seen_kw = set()
    unique_kw = []
    for kw in keywords:
        if kw not in seen_kw:
            seen_kw.add(kw)
            unique_kw.append(kw)
    air_const_codes[code] = {"description": description, "keywords": unique_kw}

output_path = REF / "air_const_codes.json"
output_path.write_text(json.dumps(air_const_codes, indent=2), encoding="utf-8")
print(f"OK: Wrote {len(air_const_codes)} AIR construction codes -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Build RMS Reference maps (flattened for TFIDF)
# ─────────────────────────────────────────────────────────────────────────────

# Build RMS Occupancy from active ATC map
atc_map_path = REF / "atc_to_air_occ_map.json"
rms_occ_codes = {}
if atc_map_path.exists():
    _atc_raw = json.loads(atc_map_path.read_text(encoding="utf-8")).get("atc_to_air", {})
    for atc_key, meta in _atc_raw.items():
        desc = meta.get("description", "")
        rms_occ_codes[atc_key] = {
            "description": desc,
            "keywords": [desc.lower()]
        }
    print(f"OK: Loaded {len(rms_occ_codes)} RMS occupancy codes for TF-IDF from atc_to_air_occ_map.")

# Build RMS Construction from active RMS map
rms_const_map_path = REF / "rms_to_air_const_map.json"
rms_const_codes = {}
if rms_const_map_path.exists():
    _rms_const_raw = json.loads(rms_const_map_path.read_text(encoding="utf-8"))
    for category in ["basic", "advanced_structural", "infrastructure", "tanks", "industrial_equipment", "chimneys", "industrial_towers", "equipment", "industrial_facilities", "special_unknown"]:
        for code, mapping in _rms_const_raw.get(category, {}).items():
            desc = mapping.get("rms_desc", "")
            keywords = mapping.get("keywords", [desc.lower()])
            # We add keywords from our alias dict (same as code_mapper does)
            _RMS_CONST_KEYWORD_ALIASES = {
                "0":  ["unknown", "unspecified", "tbd", "unclassified"],
                "1":  ["wood", "frame", "wood frame", "stick frame", "timber frame", "light frame", "light wood frame", "post frame", "pole barn", "adobe", "sip", "structural insulated panel", "modular wood"],
                "2":  ["masonry", "brick", "cmu", "block", "joisted masonry", "jm", "unreinforced masonry", "urm", "stone masonry", "concrete block", "masonry bearing", "brick masonry"],
                "3":  ["reinforced concrete", "concrete", "rc", "cast in place", "cip", "post tension", "flat slab", "waffle slab", "prestressed concrete", "concrete frame", "shear wall", "concrete shear wall"],
                "4":  ["steel", "steel frame", "structural steel", "metal frame", "light gauge steel", "lgs", "steel stud", "steel joist", "braced frame", "moment frame", "steel deck"],
                "5":  ["mobile home", "manufactured home", "trailer"],
            }
            if code in _RMS_CONST_KEYWORD_ALIASES:
                keywords.extend(_RMS_CONST_KEYWORD_ALIASES[code])
                keywords = list(dict.fromkeys(keywords))
                
            rms_const_codes[code] = {
                "description": desc,
                "keywords": keywords,
            }
    print(f"OK: Loaded {len(rms_const_codes)} RMS construction codes for TF-IDF from rms_to_air_const_map.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — abbreviations.json
# ─────────────────────────────────────────────────────────────────────────────

abbreviations = {
    "urm":  "unreinforced masonry",
    "rm":   "reinforced masonry",
    "rc":   "reinforced concrete",
    "rcc":  "reinforced concrete construction",
    "wf":   "wood frame",
    "sf":   "single family",
    "mf":   "multi family",
    "tilt": "tilt-up concrete",
    "src":  "steel reinforced concrete",
    "smrf": "steel moment resisting frame",
    "mrf":  "moment resisting frame",
    "cbf":  "concentrically braced frame",
    "ebf":  "eccentrically braced frame",
    "cmu":  "concrete masonry unit",
    "sfr":  "single family residential",
    "sfh":  "single family home",
    "mfr":  "multi family residential",
    "pc":   "precast concrete",
    "lwf":  "light wood frame",
    "bi":   "business interruption",
    "tiv":  "total insured value",
    "sqft": "square feet",
    "sqm":  "square meters",
    "yb":   "year built",
    "n/a":  "not available",
}

output_path = REF / "abbreviations.json"
output_path.write_text(json.dumps(abbreviations, indent=2), encoding="utf-8")
print(f"OK: Wrote {len(abbreviations)} abbreviations -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — iso3166_states.json  (US + Canada + key alpha-3 mappings)
# ─────────────────────────────────────────────────────────────────────────────

iso3166_states = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
    # Canada
    "Alberta": "AB", "British Columbia": "BC", "Manitoba": "MB",
    "New Brunswick": "NB", "Newfoundland and Labrador": "NL",
    "Northwest Territories": "NT", "Nova Scotia": "NS", "Nunavut": "NU",
    "Ontario": "ON", "Prince Edward Island": "PE", "Quebec": "QC",
    "Saskatchewan": "SK", "Yukon": "YT",
    # ISO alpha-3 → alpha-2 lookup
    "_alpha3_to_alpha2": {
        "USA": "US", "GBR": "GB", "CAN": "CA", "AUS": "AU", "JPN": "JP",
        "CHN": "CN", "DEU": "DE", "FRA": "FR", "MEX": "MX", "BRA": "BR",
        "IND": "IN", "ITA": "IT", "ESP": "ES", "KOR": "KR", "NLD": "NL",
        "SGP": "SG", "ZAF": "ZA", "NZL": "NZ", "IRL": "IE", "CHE": "CH",
        "NOR": "NO", "SWE": "SE", "DNK": "DK", "FIN": "FI", "BEL": "BE",
        "AUT": "AT", "PRT": "PT", "GRC": "GR", "TUR": "TR", "IDN": "ID",
        "MYS": "MY", "THA": "TH", "PHL": "PH", "VNM": "VN", "PAK": "PK",
        "BGD": "BD", "EGY": "EG", "NGA": "NG", "KEN": "KE", "GHA": "GH",
        "SAU": "SA", "ARE": "AE", "QAT": "QA", "KWT": "KW", "BHR": "BH",
        "OMN": "OM", "ISR": "IL", "POL": "PL", "CZE": "CZ", "HUN": "HU",
        "ROU": "RO", "SVK": "SK", "BGR": "BG", "HRV": "HR",
    },
}

output_path = REF / "iso3166_states.json"
output_path.write_text(json.dumps(iso3166_states, indent=2), encoding="utf-8")
print(f"OK: Wrote iso3166_states -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — iso4217_currency.json (major currencies)
# ─────────────────────────────────────────────────────────────────────────────

iso4217 = {
    "USD": "United States Dollar", "EUR": "Euro", "GBP": "Pound Sterling",
    "JPY": "Japanese Yen", "CAD": "Canadian Dollar", "AUD": "Australian Dollar",
    "CHF": "Swiss Franc", "CNY": "Chinese Yuan", "HKD": "Hong Kong Dollar",
    "SGD": "Singapore Dollar", "KRW": "South Korean Won", "INR": "Indian Rupee",
    "MXN": "Mexican Peso", "BRL": "Brazilian Real", "ZAR": "South African Rand",
    "NOK": "Norwegian Krone", "SEK": "Swedish Krona", "DKK": "Danish Krone",
    "NZD": "New Zealand Dollar", "THB": "Thai Baht", "MYR": "Malaysian Ringgit",
    "IDR": "Indonesian Rupiah", "PHP": "Philippine Peso", "VND": "Vietnamese Dong",
    "TWD": "New Taiwan Dollar", "SAR": "Saudi Riyal", "AED": "UAE Dirham",
    "QAR": "Qatari Riyal", "KWD": "Kuwaiti Dinar", "BHD": "Bahraini Dinar",
    "OMR": "Omani Rial", "ILS": "Israeli Shekel", "EGP": "Egyptian Pound",
    "PKR": "Pakistani Rupee", "BDT": "Bangladeshi Taka", "NGN": "Nigerian Naira",
    "KES": "Kenyan Shilling", "GHS": "Ghanaian Cedi", "TRY": "Turkish Lira",
    "PLN": "Polish Zloty", "CZK": "Czech Koruna", "HUF": "Hungarian Forint",
    "RON": "Romanian Leu", "BGN": "Bulgarian Lev", "HRK": "Croatian Kuna",
    "RUB": "Russian Ruble", "UAH": "Ukrainian Hryvnia", "CLP": "Chilean Peso",
    "COP": "Colombian Peso", "PEN": "Peruvian Sol", "ARS": "Argentine Peso",
}

output_path = REF / "iso4217_currency.json"
output_path.write_text(json.dumps(iso4217, indent=2), encoding="utf-8")
print(f"OK: Wrote {len(iso4217)} currency codes -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Build TF-IDF vectorizer pkl files
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_pkl(codes_dict: dict, output_pkl: pathlib.Path) -> None:
    corpus = []
    code_list = []
    for code, meta in codes_dict.items():
        desc = meta.get("description", "")
        keywords = meta.get("keywords", [])
        text = desc + " " + " ".join(keywords)
        corpus.append(text.lower())
        code_list.append(code)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vec.fit_transform(corpus)

    with open(output_pkl, "wb") as f:
        pickle.dump({"vectorizer": vec, "matrix": matrix, "codes": code_list}, f)
    print(f"OK: TF-IDF index: {len(code_list)} codes, vocab={len(vec.vocabulary_)} -> {output_pkl}")

if rms_occ_codes:
    build_tfidf_pkl(air_occ_codes,  TFIDF / "air_occ_vectorizer.pkl")
    build_tfidf_pkl(air_const_codes, TFIDF / "air_const_vectorizer.pkl")
    build_tfidf_pkl(rms_occ_codes,  TFIDF / "rms_occ_vectorizer.pkl")
    build_tfidf_pkl(rms_const_codes, TFIDF / "rms_const_vectorizer.pkl")


print("\nDone:  All reference files and TF-IDF indexes built successfully.")
print(f"    Reference files -> {REF}")
print(f"    TF-IDF caches   -> {TFIDF}")
print("\nNext: Add your API keys to .env, then run:")
print("  uvicorn main:app --reload --port 8000")
