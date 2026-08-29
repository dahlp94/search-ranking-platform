# Validation query inspection

These examples come from **project validation** only. The official test holdout was not inspected.

## bm25_strong: christian journal stickers

- query_id: `25822`
- why selected: Highest BM25 NDCG among queries with mixed ESCI labels.
- NDCG@10 BM25=1.0000, TF-IDF=1.0000, Random=0.7949

| product_id | product_title | esci_label | relevance_gain | bm25_score | tfidf_score | random_score | rank_bm25 | rank_tfidf | rank_random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B07D3Q31SH | Christian Bible Verses Scriptures Quotes Stickers (10 Sheets) - for Journal Planner Sticky Notes Scrapbooking Party Favors Decor - for Adults Men Women Kids | E | 3 | 2.400097188762412 | 0.2705319920143723 | 0.5337337470517548 | 1 | 2 | 10 |
| B07N63RB7Y | Aesthetic Planner Stickers - 1500+ Stunning Design Accessories Enhance and Simplify Your Planner, Journal, Calendar And Scrapbook | E | 3 | 2.139475453462604 | 0.2077250048583023 | 0.018231089353130447 | 2 | 4 | 16 |
| B0876CTRTF | Christian Journaling Stickers, Faith Stickers for Journals and Planners, 840+ PCS Religious Stickers, Bible Journaling Stickers, Inspirational Stickers | E | 3 | 1.5983206084474215 | 0.2746893889422912 | 0.3860747688414623 | 3 | 1 | 13 |
| B07YQPSL4P | reussir Bible Verse Stickers (Set of 48 Stickers with 24 Inspirational Christian Quotes) 1" x 2.5" Scripture Stickers | E | 3 | 1.4560970757485365 | 0.20431778379599594 | 0.8967580860241193 | 4 | 5 | 1 |
| B07XZCSFQ3 | Faith Planner Stickers \| Set of Christian Stickers for Journals, Scrapbooking, Inspirational Planners, Bible Journaling \| 135 Pieces | E | 3 | 1.3980465223490457 | 0.212543138156487 | 0.7808278464016349 | 5 | 3 | 5 |
| B085178762 | Christian Stickers for Water Bottles - 77 Inspirational Jesus Faith Stickers Pack with Bible Verse Motivational Stickers \| 77 Pieces of Religious Stickers in a Sticker Pack | E | 3 | 1.3530598055825738 | 0.1752065347470936 | 0.6747764147412492 | 6 | 6 | 8 |
| 1441329978 | Essentials Planner Stickers - Bible (Set of 450 Stickers) | E | 3 | 0.8528441360280462 | 0.12085780523283586 | 0.7204638828605924 | 7 | 8 | 7 |
| 1441330763 | Essentials Month By Month Planner Stickers (set of 475 stickers) | E | 3 | 0.8098661638187589 | 0.07847988114342587 | 0.806685185676964 | 8 | 9 | 4 |
| 1441328726 | Planner Stickers Faith | E | 3 | 0.7960758731035789 | 0.13299326258551872 | 0.4455036686808507 | 9 | 7 | 12 |
| 144132870X | Essential Weekly Planner Stickers - She Believed She Could (Set of 160 Stickers) | E | 3 | 0.771012015029853 | 0.06952583558227027 | 0.46564754376323647 | 10 | 10 | 11 |

Full ranked list: `query_25822_bm25_strong.csv`

## bm25_weak: modily clothing

- query_id: `70157`
- why selected: Lowest BM25 NDCG among queries that have at least one E/S item.
- NDCG@10 BM25=0.0000, TF-IDF=0.0000, Random=0.1781

| product_id | product_title | esci_label | relevance_gain | bm25_score | tfidf_score | random_score | rank_bm25 | rank_tfidf | rank_random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0797MNF35 | FDW Manikin 60”-67”Height Adjustable Female Dress Model Display Torso Body Tripod Stand Clothing Forms, Black | I | 0 | 0.3968747802398428 | 0.1628929745208477 | 0.17930618854613367 | 1 | 2 | 34 |
| B07HVBKYJ5 | Beige Female Dress Form Mannequin Torso Body with Black Adjustable Tripod Stand for Clothing Dress Jewelry Display | I | 0 | 0.3968747802398428 | 0.18088344644291673 | 0.9376862957773974 | 2 | 1 | 2 |
| 4805315636 | How to Create Manga: Drawing Clothing and Accessories: The Ultimate Bible for Beginning Artists (With Over 900 Illustrations) | I | 0 | 0.3866883464107866 | 0.12161915690903514 | 0.7566830825844013 | 3 | 6 | 11 |
| B07QPKR8W2 | Lovelonglong PU Leather Dog Mannequins Standing Models to Display for Dog Clothing Pet Shop Beige M (Small Dog) | I | 0 | 0.3866883464107866 | 0.10269560874513227 | 0.7236375401395351 | 4 | 12 | 12 |
| B092RM98RY | AUEAR, 5 Pack Doll Dress Form Clothing Gown Model Stands Mini Mannequin Stand for Doll Dresses Display White | I | 0 | 0.3866883464107866 | 0.1383876833939956 | 0.4620384195918823 | 5 | 5 | 19 |
| B08GLXB4ZL | Mannequin Torso Manikin Dress Form 60-67 Inch Height Adjustable Female Dress Model Display Torso Body Tripod Stand Clothing Forms, Black | I | 0 | 0.3678075916694488 | 0.15340553759671816 | 0.4131184432864329 | 6 | 3 | 24 |
| B08RP8VZHG | Tesla Model X Model S Model3 rear seat backrest headrest coat rack hook alumina clothing hanger (set of 2/black) 2017-2021 (black) | I | 0 | 0.35904215191034633 | 0.11443568829965016 | 0.06081346632445406 | 7 | 10 | 38 |
| B08VDTS61B | HEALLILY 6PCS Doll Dress Form Sewing Doll Gown Garment Mannequin Model Stand Doll Clothing Display Torso Body Support for Children Kids | I | 0 | 0.35904215191034633 | 0.1169829802151707 | 0.5273831872583087 | 8 | 8 | 18 |
| B09CMHRJRS | Mannequin Dress Form Manikin Body Dress Model, 60 Inch-67 Inch Height Adjustable with Tripod Wooden Base for Clothing Dress Jewelry Display | I | 0 | 0.35904215191034633 | 0.14257407098678793 | 0.16472135363322948 | 9 | 4 | 36 |
| B07NHN9WCS | Clothing Company and Dropshipping Bundle: Combined for a Massively Successful Business, Learn Branding, Ecommerce, Shopify, Social Media Marketing, Instagram Strategy, Graphic Design and Fashion | I | 0 | 0.33508530500171085 | 0.10131891700296249 | 0.7617804346044769 | 10 | 13 | 10 |

Full ranked list: `query_70157_bm25_weak.csv`

## brand_or_model: pilot g2 ultra fine

- query_id: `80044`
- why selected: Query contains a digit, typical of brand/model-number lookups.
- NDCG@10 BM25=1.0000, TF-IDF=1.0000, Random=0.9044

| product_id | product_title | esci_label | relevance_gain | bm25_score | tfidf_score | random_score | rank_bm25 | rank_tfidf | rank_random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B01H5U50OS | Pilot G2 Gel Ink Roller Ball Pen, Blue Ultra Fine Point, 5-COUNT (31322) | E | 3 | 2.433582833900119 | 0.32677053016624197 | 0.8634477683759851 | 1 | 5 | 7 |
| B07BHXCWZZ | Pilot G2 Retractable Premium Gel Ink Roller Ball Pens, Ultra Fine, 24 Pack, Black | E | 3 | 2.3634226817255346 | 0.38911633364124576 | 0.9888641916581075 | 2 | 2 | 1 |
| B07BHXL4NF | Pilot G2 Retractable Premium Gel Ink Roller Ball Pens, Ultra Fine, 24 Pack, Blue | E | 3 | 2.3634226817255346 | 0.3805963456283195 | 0.5079752587399526 | 3 | 3 | 19 |
| B07Q26KX2J | Pilot G2 Retractable Premium Gel Ink Roller Ball Pens Ultra Fine (36 Pack, Black) | E | 3 | 2.3634226817255346 | 0.376390590085091 | 0.31712619795874497 | 4 | 4 | 30 |
| B0017TMMLS | PILOT G2 Premium Refillable & Retractable Rolling Ball Gel Pens, Ultra Fine Point, Black Ink, 12-Pack (31277) | E | 3 | 2.2345770444996402 | 0.39107451595034265 | 0.1732455013191443 | 5 | 1 | 37 |
| B00329T22S | PILOT G2 Gel Ink Refills For Rolling Ball Pens, Ultra Fine Point, Black Ink, 2-Pack (77287) | E | 3 | 2.2345770444996402 | 0.3116593383225581 | 0.25546448193115057 | 6 | 8 | 34 |
| B0058NN8NA | PILOT G2 Premium Refillable & Retractable Rolling Ball Gel Pens, Ultra Fine Point, Black/Blue/Red/Green Inks, 4-Pack (31276) | E | 3 | 2.2345770444996402 | 0.31311810521423156 | 0.7271793187878649 | 7 | 7 | 10 |
| B00U503NE0 | PILOT G2 Gel Ink Refills For Rolling Ball Pens, Ultra Fine Point, Blue Ink, 2-Pack (77288) | E | 3 | 2.2345770444996402 | 0.3072298487704215 | 0.0324451729412405 | 8 | 9 | 40 |
| B00U503OI0 | PILOT G2 Gel Ink Refills For Rolling Ball Pens, Ultra Fine Point, Red Ink, 2-Pack (77002) | E | 3 | 2.2345770444996402 | 0.2985249252615443 | 0.49331726530514364 | 9 | 10 | 21 |
| B00AGYUBR2 | G2 Pens 31277 Black Pilot G2 Ultra-fine 0.38mm Gel Ink Pens, 1 Dozen Plus 2 Packs 0.38mm Ultra Fine Black Gel Ink Refills | E | 3 | 2.1403102829376497 | 0.22122209940350906 | 0.4106403206358138 | 10 | 12 | 28 |

Full ranked list: `query_80044_brand_or_model.csv`

## broad_generic: zodiac

- query_id: `115808`
- why selected: Short query, more likely to be broad/generic.
- NDCG@10 BM25=0.7257, TF-IDF=0.7218, Random=0.5384

| product_id | product_title | esci_label | relevance_gain | bm25_score | tfidf_score | random_score | rank_bm25 | rank_tfidf | rank_random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0091P69CI | Zodiac | E | 3 | 0.11416309772484476 | 1.0 | 0.9950265455220455 | 1 | 1 | 1 |
| B07D6ZZFZP | Zodiac | E | 3 | 0.11416309772484476 | 1.0 | 0.4822011330362397 | 2 | 2 | 14 |
| B07K82QHRY | Zodiac | I | 0 | 0.11416309772484476 | 1.0 | 0.7927029300284846 | 3 | 3 | 8 |
| B00PGV6OTY | The Zodiac Crimes | I | 0 | 0.09187170941858741 | 0.42909418520878284 | 0.26713447083162367 | 4 | 4 | 18 |
| B071RY5J7Y | Awakening The Zodiac | E | 3 | 0.09187170941858741 | 0.3951848109297064 | 0.2250215205920162 | 5 | 5 | 19 |
| B00R9O6X60 | Zodiac: Signs of the Apocalypse | E | 3 | 0.07686342344297951 | 0.2815765536057161 | 0.060438377519162256 | 6 | 6 | 25 |
| B00IG9LZ4C | Zodiac MX6 Automatic Suction Side Pool Cleaner Vacuum for Inground Pools | E | 3 | 0.05158328101575999 | 0.14880542900439142 | 0.7271930025812676 | 7 | 12 | 12 |
| B01K5BTU9U | The Zodiac, the Son of Sam, Charles Manson and Ted Bundy | I | 0 | 0.05158328101575999 | 0.1663961716053983 | 0.4642403266947861 | 8 | 7 | 15 |
| 1365885739 | America's Jack The Ripper: The Crimes and Psychology of the Zodiac Killer | E | 3 | 0.04648682348909617 | 0.15675736852217248 | 0.038146069374602964 | 9 | 9 | 27 |
| B0786RL7RC | The Hunt for Zodiac: The Inconceivable Double Life of a Notorious Serial Killer | E | 3 | 0.04648682348909617 | 0.15957072280315174 | 0.12703772246123002 | 10 | 8 | 23 |

Full ranked list: `query_115808_broad_generic.csv`

## lexical_overlap_misleading: apple iphone 11 pro unlocked

- query_id: `9724`
- why selected: BM25 top result has high title overlap but is labeled Irrelevant.
- NDCG@10 BM25=0.5563, TF-IDF=0.6624, Random=0.5651

| product_id | product_title | esci_label | relevance_gain | bm25_score | tfidf_score | random_score | rank_bm25 | rank_tfidf | rank_random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B07Z4681LQ | (Renewed) Apple iPhone 11 Pro Max, US Version, 256GB, Midnight - Unlocked | I | 0 | 2.198713461203979 | 0.5860498034558295 | 0.6265354744348411 | 1 | 2 | 7 |
| B07ZQRL9XY | Apple iPhone 11 Pro, US Version, 256GB, Space Gray - Unlocked (Renewed) | E | 3 | 2.198713461203979 | 0.6404364331956389 | 0.6182187116116976 | 2 | 1 | 8 |
| B07ZPKR714 | (Renewed) Apple iPhone 11, US Version, 128GB, Black - Unlocked | I | 0 | 1.852643123192435 | 0.40789956740726474 | 0.7979274729466095 | 3 | 3 | 5 |
| B07XSRG2BZ | Apple Simple Mobile Prepaid - Apple iPhone 11 Pro (64GB) - Midnight Green [Locked to Carrier – Simple Mobile] | S | 2 | 1.6631881446371075 | 0.28424542054341695 | 0.01164906015842071 | 4 | 5 | 16 |
| B07XSR733W | Simple Mobile Prepaid - Apple iPhone 11 Pro Max (64GB) - Space Gray [Locked to Carrier – Simple Mobile] | I | 0 | 1.4979926671899628 | 0.2632956528268009 | 0.9060755237229866 | 5 | 6 | 2 |
| B0775H5HJW | Apple iPhone X, GSM Unlocked, 256GB - Silver (Renewed) | S | 2 | 1.3768491965985599 | 0.23260953352201463 | 0.10517062322111459 | 6 | 8 | 13 |
| B082MDD4BD | [3 Pack] QHOHQ Camera Lens Protector for iPhone 11 Pro Max(6.5"),iPhone 11 Pro(5.8") Tempered Glass,[Easy to Install] [9H Hardness] Anti-Scratch Screen Protector (Black) | C | 1 | 1.3600258919465578 | 0.3142728133202841 | 0.971275553759025 | 7 | 4 | 1 |
| B07756QYST | (Renewed) Apple iPhone 8, US Version, 64GB, Silver - Unlocked | S | 2 | 1.3177181842781924 | 0.2404644598215398 | 0.5086303693364302 | 8 | 7 | 9 |
| B07PBCJWJ4 | Apple iPhone XR, US Version, 256GB, Red - Unlocked (Renewed) | S | 2 | 1.3177181842781924 | 0.21244662823927388 | 0.3365649190028105 | 9 | 12 | 11 |
| B081TJFVCJ | Apple iPhone X, 64GB, Space Gray - Fully Unlocked (Renewed) | S | 2 | 1.3177181842781924 | 0.22663467974991103 | 0.2620559659141427 | 10 | 10 | 12 |

Full ranked list: `query_9724_lexical_overlap_misleading.csv`

## tfidf_bm25_disagree: ugly stik custom

- query_id: `106401`
- why selected: Largest absolute NDCG@10 gap between BM25 and TF-IDF.
- NDCG@10 BM25=0.1997, TF-IDF=0.8391, Random=0.6651

| product_id | product_title | esci_label | relevance_gain | bm25_score | tfidf_score | random_score | rank_bm25 | rank_tfidf | rank_random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B00A0PA3R0 | Wild River CLC WT3505 Tackle Tek Mission Lighted Convertible Tackle Bag with Four PT3500 Trays, Small | I | 0 | 0.0 | 0.0 | 0.8871212862126845 | 1 | 9 | 3 |
| B00A0PA4R4 | Wild River by CLC WT3503 Tackle Tek Recon Lighted Compact Tackle Backpack & Four PT3500 Trays, Clear, Water-Resistant Phone Storage | I | 0 | 0.0 | 0.0 | 0.5318567352239485 | 2 | 10 | 6 |
| B00F66Z2XS | Wild River CLC WCN503 Tackle Tek Recon LED Lighted Camo Compact Backpack without Trays, Mossy Oak | I | 0 | 0.0 | 0.0 | 0.22728555598367717 | 3 | 11 | 11 |
| B00FH3WNY6 | Shakespeare USSP662M/35CBO Ugly Stik GX2 2-Piece Fishing Rod and Spinning Reel Combo, 6 Feet 6 Inch, Medium Power | E | 3 | 0.0 | 0.1921857609507678 | 0.03629264508908425 | 4 | 8 | 15 |
| B00QQUEGOO | Wild River CLC WN3508 Multi-Tackle Small Backpack without Trays, Beige | I | 0 | 0.0 | 0.0 | 0.23095172164542788 | 5 | 12 | 10 |
| B00QROOCRU | Wild River CLC WT3606 Multi-Tackle Large Backpack with Two 3600 Style Trays | I | 0 | 0.0 | 0.0 | 0.004336370325093153 | 6 | 13 | 16 |
| B00T5VQ7WK | Wild River Nomad CLC WCN604 Tackle Tek Nomad LED Lighted Camo Backpack, Mossy Oak | I | 0 | 0.0 | 0.0 | 0.43231359512236467 | 7 | 14 | 8 |
| B011LUVJWU | Ugly Stik GX2 Baitcast Combo | E | 3 | 0.0 | 0.4212255583764613 | 0.2568302331740826 | 8 | 6 | 9 |
| B011ODA4W0 | Shakespeare Ugly Stik Logo Hoodie | I | 0 | 0.0 | 0.4511770780213013 | 0.682348771331445 | 9 | 4 | 5 |
| B011ODOFIY | Shakespeare Ugly Stik Logo Hoodie | I | 0 | 0.0 | 0.4511770780213013 | 0.04910368473031934 | 10 | 5 | 14 |

Full ranked list: `query_106401_tfidf_bm25_disagree.csv`
