"""Generate a small fake dataset in the challenge format, for smoke tests only.

It mimics the documented noise (abbreviations, suffix changes, typos, word
swaps, DBA names, landmarks, missing postcodes) and adds hard negatives
(same brand at another branch, near-identical names elsewhere). Train has
US + India, test adds France. Real scores will differ; use it to check that
the pipeline runs and the output format is valid.
"""
import argparse
import random
from pathlib import Path

R = random.Random(7)

WORDS = {
    "US": ["Summit", "Liberty", "Eagle", "Pioneer", "Golden", "Harbor", "Maple",
           "Granite", "Blue", "Ridge", "Frontier", "Oak", "Silver", "Prairie",
           "Atlas", "Keystone", "Beacon", "Cedar", "Union", "Evergreen"],
    "India": ["Lakshmi", "Shree", "Ganesh", "Sai", "Balaji", "Krishna", "Durga",
              "Annapurna", "Bharat", "Om", "Vinayak", "Jai", "Mahalaxmi",
              "Sharma", "Patel", "Agarwal", "Reddy", "Kaveri", "Ganga", "Surya"],
    "France": ["Dupont", "Lumiere", "Beaulieu", "Moreau", "Etoile", "Rousseau",
               "Chateau", "Belle", "Martin", "Soleil", "Provence", "Lefevre",
               "Riviere", "Fontaine", "Mercier", "Clair", "Durand", "Vallee"],
}
TRADES = {
    "US": ["Plumbing", "Dental", "Logistics", "Auto Repair", "Bakery", "Consulting",
           "Construction", "Insurance", "Electric", "Pharmacy", "Realty"],
    "India": ["Traders", "Sweets", "Textiles", "Electronics", "Jewellers",
              "Hardware", "Enterprises", "Medicals", "Agencies", "Motors"],
    "France": ["Boulangerie", "Pharmacie", "Transports", "Garage", "Conseil",
               "Menuiserie", "Immobilier", "Traiteur", "Coiffure", "Plomberie"],
}
SUFFIX = {"US": [("Inc", "Incorporated"), ("LLC", "LLC"), ("Corp", "Corporation"),
                 ("Co", "Company")],
          "India": [("Pvt Ltd", "Private Limited"), ("Ltd", "Limited"), ("", "")],
          "France": [("SARL", "SARL"), ("SAS", "SAS"), ("SA", "SA"), ("", "")]}
STREETS = {"US": [("St", "Street"), ("Ave", "Avenue"), ("Rd", "Road"),
                  ("Blvd", "Boulevard")],
           "India": [("Rd", "Road"), ("Marg", "Marg"), ("Nagar", "Nagar"),
                     ("St", "Street")],
           "France": [("Av", "Avenue"), ("Bd", "Boulevard"), ("Rue", "Rue"),
                      ("Pl", "Place")]}
CITIES = {"US": [("Austin", "TX"), ("Denver", "CO"), ("Columbus", "OH"),
                 ("Portland", "OR"), ("Tampa", "FL")],
          "India": [("Pune", "Maharashtra"), ("Jaipur", "Rajasthan"),
                    ("Indore", "Madhya Pradesh"), ("Kochi", "Kerala")],
          "France": [("Lyon", ""), ("Marseille", ""), ("Toulouse", ""),
                     ("Bordeaux", ""), ("Lille", "")]}
TRANSLIT = [("aksh", "ax"), ("ee", "i"), ("sh", "s"), ("v", "w"), ("aa", "a"),
            ("oo", "u"), ("th", "t")]
LANDMARKS = ["Near SBI ATM", "Opp Bus Stand", "Behind City Mall", "Near Railway Station"]


def typo(s):
    if len(s) < 5 or R.random() < 0.5:
        return s
    i = R.randrange(1, len(s) - 1)
    op = R.choice("dsr")
    if op == "d":
        return s[:i] + s[i + 1:]
    if op == "s":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + R.choice("aeiourstn") + s[i + 1:]


def make_entity(country, eid):
    w = WORDS[country]
    core = " ".join(R.sample(w, R.choice([1, 2])))
    trade = R.choice(TRADES[country])
    suf = R.choice(SUFFIX[country])
    street = R.choice(STREETS[country])
    city, state = R.choice(CITIES[country])
    num = R.randint(1, 999)
    pc = {"US": f"{R.randint(10000, 99999)}", "India": f"{R.randint(110000, 859999)}",
          "France": f"{R.randint(10000, 95999)}"}[country]
    sname = R.choice(w + ["Main", "Park", "Station", "Market", "Church"])
    return dict(eid=eid, country=country, core=core, trade=trade, suf=suf,
                street=street, sname=sname, num=num, city=city, state=state, pc=pc)


def render(e, noisy):
    long_ = not noisy or R.random() < 0.5
    suf = e["suf"][1] if long_ else e["suf"][0]
    if noisy and R.random() < 0.3:
        suf = ""
    name = f"{e['core']} {e['trade']} {suf}".strip()
    street = e["street"][1] if not noisy or R.random() < 0.5 else e["street"][0]
    if e["country"] == "France":
        addr_parts = [f"{e['num']} {street} {e['sname']}", f"{e['pc']} {e['city']}"]
    else:
        addr_parts = [f"{e['num']} {e['sname']} {street}", e["city"],
                      f"{e['state']} {e['pc']}".strip()]
    if noisy:
        if R.random() < 0.3:
            name = name.replace(" and ", " & ")
            toks = name.split()
            if len(toks) > 2 and R.random() < 0.4:
                toks[0], toks[1] = toks[1], toks[0]
            name = " ".join(toks)
        name = typo(typo(name))
        if R.random() < 0.2:
            addr_parts[0] = addr_parts[0].replace(f"{e['num']} ", "", 1)
        elif R.random() < 0.2:
            addr_parts[0] = addr_parts[0].replace(f"{e['num']} ", f"No. {e['num']}, ", 1)
        if R.random() < 0.3:
            addr_parts[0] = typo(typo(addr_parts[0]))
        if e["country"] == "India" and R.random() < 0.4:
            a, b = R.choice(TRANSLIT)
            name = name.replace(a, b)
        if R.random() < 0.1:
            name = f"{name} DBA {e['core'].split()[0]} {e['trade']}"
        if R.random() < 0.25:
            addr_parts[-1] = addr_parts[-1].replace(e["pc"], "").strip()
        if e["country"] == "India" and R.random() < 0.3:
            addr_parts.insert(1, R.choice(LANDMARKS))
        if R.random() < 0.15:
            addr_parts = addr_parts[:1]
        if R.random() < 0.2:
            name = name.upper()
    if e["country"] == "France" and R.random() < 0.5:
        name = name.replace("Etoile", "Étoile").replace("Riviere", "Rivière")
    return name, ", ".join(p for p in addr_parts if p)


def build(countries, n_per_country):
    s1, s2, s3, gt = [], [], [], {}
    c1 = c2 = c3 = 0
    for country in countries:
        for _ in range(n_per_country):
            c1 += 1
            e = make_entity(country, f"S1-{c1:06d}")
            s1.append((e["eid"], *render(e, False), country))
            matches = []
            if R.random() < 0.75:           # 25% singletons
                for src, bucket in ((2, s2), (3, s3)):
                    for _ in range(R.choice([0, 1, 1, 1, 2])):
                        if src == 2:
                            c2 += 1
                            rid = f"S2-{c2:06d}"
                        else:
                            c3 += 1
                            rid = f"S3-{c3:06d}"
                        bucket.append((rid, *render(e, True), country))
                        matches.append(rid)
            gt[e["eid"]] = matches
            if R.random() < 0.3:            # hard negative: same brand, other branch
                f = dict(e)
                f["num"], f["pc"] = R.randint(1, 999), str(int(e["pc"]) + R.randint(1, 50))
                f["sname"] = R.choice(WORDS[country])
                c2 += 1
                s2.append((f"S2-{c2:06d}", *render(f, True), country))
            if R.random() < 0.3:            # hard negative: different trade, same core
                f = make_entity(country, "x")
                f["core"] = e["core"]
                c3 += 1
                s3.append((f"S3-{c3:06d}", *render(f, True), country))
            if R.random() < 0.3:            # hard negative: other business, same building
                f = make_entity(country, "x")
                for k in ("street", "sname", "num", "city", "state", "pc"):
                    f[k] = e[k]
                f["trade"] = e["trade"] if R.random() < 0.5 else f["trade"]
                c2 += 1
                s2.append((f"S2-{c2:06d}", *render(f, True), country))
            if R.random() < 0.15:           # hard negative: near-identical name, one letter off
                f = dict(e)
                f["core"] = typo(typo(e["core"])) + R.choice(["", " New"])
                f["city"], f["state"] = R.choice(CITIES[country])
                f["pc"], f["num"] = str(int(e["pc"]) + 7), R.randint(1, 999)
                c3 += 1
                s3.append((f"S3-{c3:06d}", *render(f, True), country))
    return s1, s2, s3, gt


def write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_dataset")
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args()
    out = Path(args.out)
    for split, countries in (("train", ["US", "India"]), ("test", ["US", "India", "France"])):
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        s1, s2, s3, gt = build(countries, args.n if split == "train" else args.n // 2)
        R.shuffle(s2)
        R.shuffle(s3)
        for k, rows in ((1, s1), (2, s2), (3, s3)):
            write(d / f"{split}_source{k}.tsv", rows)
        if split == "train":
            with open(d / "train_ground_truth.tsv", "w") as fh:
                fh.write("source1_entity_id\tmatched_entity_ids\n")
                for sid, ms in gt.items():
                    fh.write(f"{sid}\t{','.join(ms)}\n")
        else:
            with open(out / "test_ground_truth_HIDDEN.tsv", "w") as fh:
                fh.write("source1_entity_id\tmatched_entity_ids\n")
                for sid, ms in gt.items():
                    fh.write(f"{sid}\t{','.join(ms)}\n")
    print(f"wrote synthetic data to {out}/")


if __name__ == "__main__":
    main()
