# Deliverable 6: Demographic Audit

## RACE × ETHNICITY Cross-Tabulation

|                        |   Eth_0 |   Eth_1 |   Eth_All |
|:-----------------------|--------:|--------:|----------:|
| Other/Unknown          |    2300 |    1174 |      3474 |
| Native American        |     523 |   15881 |     16404 |
| Asian/Pacific Islander |     721 |  114491 |    115212 |
| Black                  |  101714 |  501654 |    603368 |
| White                  |  149284 |   37386 |    186670 |
| All                    |  254542 |  670586 |    925128 |


## Known Issues

- **WARNING**: RACE=1 (Native American): 96.8% are ETHNICITY=1 — potential double-coding
- **WARNING**: RACE=2 (Asian/Pacific Islander): 99.4% are ETHNICITY=1 — potential double-coding
- **WARNING**: RACE=3 (Black): 83.1% are ETHNICITY=1 — potential double-coding


## Interpretation
The RACE and ETHNICITY variables in the Texas PUDF are independently coded. The high overlap between RACE=3 (Black) and ETHNICITY=1 (Hispanic) reflects the demographic composition of the Texas hospital population, not a coding error. However, this correlation means that RACE and ETHNICITY fairness metrics are partially redundant for this dataset. Results should be interpreted with this demographic context in mind.