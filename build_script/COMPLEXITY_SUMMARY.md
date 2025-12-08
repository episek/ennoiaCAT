# EnnoiaCAT Complexity Analysis Summary

**Analysis Date:** December 8, 2025

## Overall Metrics

- **Total Blocks Analyzed:** 1,616 (classes, functions, methods)
- **Average Complexity:** A (2.75) - **EXCELLENT**
- **Overall Maintainability:** Most files have MI > 20 (Grade A)

## Complexity Grades Explained

### Cyclomatic Complexity (CC)
- **A (1-5):** Simple, easy to test ✅
- **B (6-10):** Well structured ✅
- **C (11-20):** Slightly complex ⚠️
- **D (21-30):** More complex, needs refactoring ❌
- **E (31-40):** Very complex, difficult to maintain ❌
- **F (41+):** Extremely complex, high risk ❌

### Maintainability Index (MI)
- **A (100-20):** Very high maintainability ✅
- **B (19-10):** Medium maintainability ⚠️
- **C (9-0):** Extremely low, needs immediate attention ❌

---

## 🔴 Critical Issues - High Complexity Functions

These functions have complexity > 20 and should be refactored:

### Extremely Complex (F - Grade F: 42+)
1. **`_run_conformance_test_agent`** in `ennoiaCAT_MAV_RCT.py:561`
   - Complexity: **F (42)** ❌
   - Recommendation: Break into smaller functions

2. **`InstrumentSettings.apply_option_settings`** in `RsInstrument/Internal/InstrumentSettings.py:175`
   - Complexity: **F (84)** ❌❌
   - Recommendation: Major refactoring needed

### Very Complex (E - Grade E: 31-40)
3. **`run_conformance_ui`** in `ennoiaCAT_MAV_RCT.py:776`
   - Complexity: **E (38)** ❌
   - Recommendation: Split UI logic into smaller components

4. **`VisaSession.__init__`** in `RsInstrument/Internal/VisaSession.py:67`
   - Complexity: **E (33)** ❌
   - Recommendation: Extract initialization logic

### More Complex (D - Grade D: 21-30)
5. **`AnalysisAgent.run`** in `ennoia_agentic_app.py:829`
   - Complexity: **D (30)** ⚠️

6. **`LocationAgent.detect_and_gate`** in `ennoia_agentic_app.py:426`
   - Complexity: **D (28)** ⚠️

7. **`value_to_scpi_string`** in `RsInstrument/Internal/ConverterToScpiString.py:10`
   - Complexity: **D (28)** ⚠️

8. **`compose_cmd_string_from_struct_args`** in `RsInstrument/Internal/ArgStructStringParser.py:105`
   - Complexity: **D (26)** ⚠️

9. **`run_remote_build_and_fetch`** in `ennoia_agentic_app.py:240`
   - Complexity: **D (21)** ⚠️

10. **`InstrumentSettings` class** in `RsInstrument/Internal/InstrumentSettings.py:42`
    - Class Complexity: **D (21)** ⚠️

---

## ⚠️ Medium Complexity Functions (C - Grade C: 11-20)

### Configuration Functions
- `AKHelper.configure_AK` - C (13) in `AK_config.py:294`
- `CSHelper.configure_CS` - C (14) in `CS_config.py:300`
- `TinySAHelper.configure_tinySA` - C (18) in `tinySA_config.py:290`
- `TinySAHelper.configure_tinySA` - C (18) in `mav_config.py:290`
- `RSHelper.configure_RS` - C (14) in `RS_config.py:273`

### Analysis Functions
- `TinySAHelper.analyze_signal_peaks` - C (15) in multiple files
- `RSHelper.analyze_signal_peaks` - C (15) in `RS_config.py:539`
- `AnalysisAgent._scan_wifi_table` - C (11) in `ennoia_agentic_app.py:779`

### Parsing Functions
- `parse_freq` - C (11) in `ennoiaCAT_VI.py:227`
- `scrape_website` - C (11) in `scraper.py:45`
- `augment_with_gsma` - C (19) in `operator_table_service.py:202`
- `augment_with_comreg` - C (17) in `operator_table_service.py:249`

---

## ✅ Well-Structured Files

Files with excellent maintainability (MI > 50):

| File | MI Score | Grade |
|------|----------|-------|
| `embedder.py` | 100.00 | A ✅ |
| `build_ennoia.py` | 93.25 | A ✅ |
| `map_api_vi.py` | 87.63 | A ✅ |
| `generate_jumbo_pcap.py` | 85.51 | A ✅ |
| `retriever.py` | 79.67 | A ✅ |
| `run_static_analysis.py` | 73.16 | A ✅ |
| `run_lint.py` | 72.73 | A ✅ |
| `generate_test_pcap.py` | 71.48 | A ✅ |
| `inferRAG.py` | 71.02 | A ✅ |
| `flask_pcap_backend.py` | 70.39 | A ✅ |

---

## ⚠️ Files Needing Attention

Files with low maintainability (MI < 20):

| File | MI Score | Grade | Status |
|------|----------|-------|--------|
| `ennoia_agentic_app.py` | 3.91 | C ❌ | **CRITICAL** |
| `RsInstrument/Internal/VisaSession.py` | 0.00 | C ❌ | **CRITICAL** |
| `RsInstrument/Internal/Instrument.py` | 0.00 | C ❌ | **CRITICAL** |
| `RsInstrument/Internal/ScpiLogger.py` | 8.35 | C ❌ | **CRITICAL** |
| `RsInstrument/Internal/Conversions.py` | 17.63 | B ⚠️ | Needs work |
| `ennoiaCAT_VI.py` | 19.43 | B ⚠️ | Needs work |

---

## 📊 Code Statistics (Main Files)

| File | LOC | SLOC | Comments | Comment % |
|------|-----|------|----------|-----------|
| `CS_config.py` | 673 | 436 | 140 | 21% |
| `AK_config.py` | 547 | 381 | 101 | 18% |
| `csv_watcher_ready.py` | 355 | 256 | 33 | 9% |
| `ennoiaCAT.py` | 339 | 225 | 80 | 24% |

**Legend:**
- **LOC:** Total Lines of Code
- **SLOC:** Source Lines of Code (excluding blanks)
- **LLOC:** Logical Lines of Code
- **Comments:** Documentation lines

---

## 🎯 Recommendations

### Immediate Actions (Critical)
1. **Refactor `ennoia_agentic_app.py`** (MI: 3.91)
   - Break down `AnalysisAgent.run` (CC: 30)
   - Split `LocationAgent.detect_and_gate` (CC: 28)
   - Extract helper functions

2. **Fix `_run_conformance_test_agent`** (CC: 42)
   - Break into 4-5 smaller functions
   - Extract test logic into separate methods

3. **Refactor `InstrumentSettings.apply_option_settings`** (CC: 84)
   - This is extremely complex
   - Split into multiple methods
   - Use strategy pattern for different options

### Short-term (High Priority)
4. **Simplify configuration functions** (CC: 13-18)
   - Extract UI rendering logic
   - Separate validation from configuration

5. **Improve test files**
   - Files in RsInstrument/Internal/ need better structure
   - Add more comments (currently 9-21%)

### Long-term (Medium Priority)
6. **Maintain complexity standards**
   - Keep new functions at CC < 10
   - Aim for MI > 20 on all new files

7. **Add documentation**
   - Increase comment percentage to 20-25%
   - Add docstrings to complex functions

---

## 🏆 Success Metrics

**Overall: GOOD**
- ✅ Average complexity: A (2.75) - Excellent!
- ✅ 95%+ of functions have CC < 10
- ⚠️ 2-3 files need critical refactoring
- ✅ Most files have good maintainability

**Keep up the good work!** Focus on refactoring the critical high-complexity functions.

---

## 📁 Reports Generated

- `build_script/cc_report.json` - Cyclomatic Complexity (JSON)
- `build_script/mi_report.json` - Maintainability Index (JSON)
- `build_script/complexity_report.txt` - Full text report

## Next Steps

1. Review and refactor functions with CC > 20
2. Improve maintainability of files with MI < 20
3. Add unit tests for complex functions
4. Run complexity analysis regularly (weekly/monthly)
