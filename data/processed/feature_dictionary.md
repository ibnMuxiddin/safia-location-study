# Xususiyatlar lug'ati (`hex_features.parquet`)

Bir qator = bitta H3 katak (resolution 8; Toshkentda ~0.88 km², markazdan chekkasigacha ~500 m). Hudud: 12 tuman, Yangi Toshkent kiritilmagan.

| Ustun | Ma'no | Modelda |
|---|---|---|
| `h3`, `lat`, `lon` | katak id va markazi | yo'q |
| `cv_block` | H3 resolution 6 ota-katak, fazoviy CV guruhi | guruh |
| `district` | katak markazi tushgan tuman | yo'q |
| `n_university` | OSM `university` obyektlari soni, katak ichida | ha |
| `n_university_k1` | OSM `university` obyektlari soni, katak + 6 qo'shni | ha |
| `n_school` | OSM `school` obyektlari soni, katak ichida | ha |
| `n_school_k1` | OSM `school` obyektlari soni, katak + 6 qo'shni | ha |
| `n_metro` | OSM `metro` obyektlari soni, katak ichida | ha |
| `n_metro_k1` | OSM `metro` obyektlari soni, katak + 6 qo'shni | ha |
| `n_bus_stop` | OSM `bus_stop` obyektlari soni, katak ichida | ha |
| `n_bus_stop_k1` | OSM `bus_stop` obyektlari soni, katak + 6 qo'shni | ha |
| `n_office` | OSM `office` obyektlari soni, katak ichida | ha |
| `n_office_k1` | OSM `office` obyektlari soni, katak + 6 qo'shni | ha |
| `n_mall` | OSM `mall` obyektlari soni, katak ichida | ha |
| `n_mall_k1` | OSM `mall` obyektlari soni, katak + 6 qo'shni | ha |
| `n_supermarket` | OSM `supermarket` obyektlari soni, katak ichida | ha |
| `n_supermarket_k1` | OSM `supermarket` obyektlari soni, katak + 6 qo'shni | ha |
| `n_hospital` | OSM `hospital` obyektlari soni, katak ichida | ha |
| `n_hospital_k1` | OSM `hospital` obyektlari soni, katak + 6 qo'shni | ha |
| `n_apartments` | OSM `apartments` obyektlari soni, katak ichida | ha |
| `n_apartments_k1` | OSM `apartments` obyektlari soni, katak + 6 qo'shni | ha |
| `n_cafe` | OSM `cafe` obyektlari soni (Safia va raqobatchi brendlar kafelari chiqarilgan — leakage), katak ichida | ha |
| `n_cafe_k1` | OSM `cafe` obyektlari soni (Safia va raqobatchi brendlar kafelari chiqarilgan — leakage), katak + 6 qo'shni | ha |
| `n_apartment_levels` | ko'p qavatli uylar qavatlari yig'indisi, katak ichida | ha |
| `n_apartment_levels_k1` | ko'p qavatli uylar qavatlari yig'indisi, katak + 6 qo'shni | ha |
| `district_density` | tumanning rasmiy aholi zichligi, kishi/km² (2023-04-01) | ha |
| `dist_centre_km` | Amir Temur xiyobonigacha masofa, km | ha |
| `nonres_share` | katakning noturar hudud (sanoat, aeroport, qabriston...) bilan qoplangan ulushi | ha (07-qadamda qo'shilgan) |
| `competitors_1km` | markazdan 1000 m ichidagi raqobatchilar soni (Bon!, Cake Lab, Breadly) | faqat tushuntirish uchun |
| `has_competitor_1km` | 1000 m ichida raqobatchi bor (1/0); Bon!/Breadly to'liq emas | ha |
| `has_safia` | **target**: katakda oddiy Safia filiali bor (1/0) | target |
| `safia_count` | katakdagi oddiy filiallar soni | **yo'q — leakage** |
| `dist_safia_m` | markazdan eng yaqin oddiy Safia'gacha, m | **yo'q — leakage**, faqat 10-qadamda |
