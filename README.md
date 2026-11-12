Projekt 1. Wskaźnik MACD – instrukcja uruchomienia

Autor: Franciszek Łaba
Przedmiot: Metody Numeryczne (termin P3 – część podstawowa i eksperymentalna)

Program wykonuje symulację strategii MACD(12, 26, 9) dla części podstawowej (AAPL, opóźnienie d=1) oraz analizę porównawczą zmodyfikowanego algorytmu MACD(12, 26, 9) z filtrem wykupienia RSI(14) < 70 na zbiorze 117 spółek Nasdaq-100 (dane_pro1.zip). Wszystkie wykresy generowane są w formacie wektorowym PDF.


1. Szybki start

Wszystkie polecenia wykonuje się w katalogu głównym projektu (MACD_indicator_simpulation/):

# 1. Utworzenie środowiska i instalacja wymaganych pakietów (pandas, numpy, matplotlib)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Część podstawowa – wykresy A1, B1, C1, C2, D1, E1 i E2 oraz statystyki do slajdu F
python main.py podstawowa

# 3. Część eksperymentalna – wykresy G1, G2, H1, H2 i statystyki dla zbioru dane_pro1.zip
python main.py eksperyment --dane /sciezka/do/katalogu_z_csv

# (Opcjonalnie) Jedno polecenie generujące wszystkie wykresy obu części (gdy CSV są w ./data):
python main.py wszystko


2. Gdzie umieścić dane wejściowe

- Część podstawowa: plik dane_podstawowe/AAPL.csv (L = 1005 sesji dziennych Apple Inc. w USD z lat 2021–2024) jest dołączony do projektu i wczytywany domyślnie. Własny plik CSV można podać flagą --plik SCIEZKA.
- Część eksperymentalna: zgodnie z p. 6.3 instrukcji zbiór dane_pro1.zip nie jest dołączany do archiwum ZIP z kodem. Rozpakowane pliki *.csv ze zbioru dane_pro1.zip można przekazać na trzy sposoby:
  1. skopiować pliki *.csv do katalogu ./data/ wewnątrz projektu (lokalizacja domyślna),
  2. wskazać katalog przy uruchomieniu: python main.py eksperyment --dane /sciezka/do/dane_pro1,
  3. ustawić zmienną środowiskową: export MACD_DATA_DIR=/sciezka/do/dane_pro1.


3. Polecenia generujące poszczególne wykresy

- Wykresy A1, B1, C1, C2, D1, E1 i E2 (część podstawowa):
  python main.py podstawowa

  Wygenerowane pliki PDF zapisywane są w wyniki/wykresy/ (oraz kopiowane do presentation/figures/):
  - A1_cena.pdf – wykres A1 (cena zamknięcia dla wszystkich L próbek z okresem rozruchu 0..100),
  - B1_cena_macd.pdf – wykres B1 (cena w tle na lewej osi Y, MACD i SIGNAL na prawej osi Y),
  - C1_C2_sygnaly_histogram.pdf – wykresy C1 i C2 (MACD i SIGNAL z sygnałami od próbki 101 oraz histogram H_i),
  - D1_transakcja.pdf – wykres D1 (powiększenie wybranej zamkniętej transakcji z momentami sygnału i wykonania),
  - E1_E2_symulacja.pdf – wykresy E1 i E2 (cena z sygnałami wykonanymi/niewykonanymi oraz wartość portfela V_i od próbki 100),
  - wyniki/podstawowa_podsumowanie.txt – zestawienie statystyk końcowych ze slajdu F.

- Wykresy części eksperymentalnej (G1, G2, H1, H2):
  python main.py eksperyment --dane data

  Zapisuje w wyniki/wykresy/: G1_wynik_globalny.pdf (wynik globalny), G2_analiza_filtru.pdf (wrażliwość na próg RSI), H1_przyklad_lokalny.pdf (przykład poprawy – QCOM) oraz H2_przyklad_lokalny.pdf (przykład pogorszenia – SMCI).

- Wszystkie wykresy jednocześnie (A1–E2 oraz G1–H2):
  python main.py wszystko

- Testy jednostkowe:
  python -m unittest discover -s tests -v


4. Kompilacja prezentacji końcowej (LaTeX Beamer)

cd presentation
make           # dwukrotne uruchomienie pdflatex -> prezentacja.pdf
make clean     # usunięcie plików pomocniczych LaTeX
make distclean # usunięcie plików pomocniczych oraz prezentacja.pdf


5. Struktura katalogów

- main.py – interfejs wiersza poleceń (CLI),
- src/ – moduły obliczeniowe i wizualizacyjne (data_loader.py, indicators.py, strategies.py, backtester.py, metrics.py, experiment.py, visualizer.py, pipeline.py),
- dane_podstawowe/AAPL.csv – dane wejściowe do części podstawowej,
- tests/ – testy jednostkowe (test_indicators.py, test_backtester.py),
- presentation/ – kod źródłowy prezentacji końcowej (prezentacja.tex), Makefile oraz katalog figures/.
