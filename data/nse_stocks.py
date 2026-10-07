"""
Comprehensive list of NSE equity tickers (yfinance format with .NS suffix).
Covers Nifty 50, Nifty Next 50, Nifty Midcap 150, Nifty Smallcap 250, and more.
"""

import csv
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import requests

# --- NIFTY 50 ---
NIFTY_50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BEL.NS", "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "ETERNAL.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "ITC.NS", "INDUSINDBK.NS", "INFY.NS",
    "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SHRIRAMFIN.NS", "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS",
    "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]

# --- NIFTY NEXT 50 ---
NIFTY_NEXT_50 = [
    "ABB.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS",
    "ATGL.NS", "BANKBARODA.NS", "BOSCHLTD.NS", "CANBK.NS",
    "CHOLAFIN.NS", "COLPAL.NS", "DLF.NS", "DABUR.NS",
    "DMART.NS", "GODREJCP.NS", "HAVELLS.NS", "HAL.NS",
    "ICICIPRULI.NS", "ICICIGI.NS", "INDHOTEL.NS", "IOC.NS",
    "IRFC.NS", "IRCTC.NS", "JIOFIN.NS", "JINDALSTEL.NS",
    "JSWENERGY.NS", "LICI.NS", "LTIM.NS", "MANKIND.NS",
    "MARICO.NS", "MAXHEALTH.NS", "NHPC.NS", "NAUKRI.NS",
    "PFC.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS",
    "SBICARD.NS", "SRF.NS", "MOTHERSON.NS", "SIEMENS.NS",
    "TATAELXSI.NS", "TATAPOWER.NS", "TORNTPHARM.NS", "TVSMOTOR.NS",
    "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "ZOMATO.NS",
    "ZYDUSLIFE.NS",
]

# --- NIFTY MIDCAP 150 (selected) ---
NIFTY_MIDCAP = [
    "AARTIIND.NS", "ACC.NS", "ABCAPITAL.NS", "ABFRL.NS",
    "AJANTPHARM.NS", "ALKEM.NS", "APLLTD.NS", "ASTRAL.NS",
    "AUROPHARMA.NS", "BALKRISIND.NS", "BANDHANBNK.NS", "BATAINDIA.NS",
    "BERGEPAINT.NS", "BHEL.NS", "BIOCON.NS", "CANFINHOME.NS",
    "CENTRALBK.NS", "CESC.NS", "CGPOWER.NS", "CHAMBLFERT.NS",
    "COFORGE.NS", "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS",
    "CUB.NS", "CUMMINSIND.NS", "DEEPAKNTR.NS", "DELTACORP.NS",
    "DEVYANI.NS", "DIVISLAB.NS", "DIXON.NS", "EMAMILTD.NS",
    "ENDURANCE.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS",
    "FORTIS.NS", "GAIL.NS", "GLS.NS", "GLENMARK.NS",
    "GMRINFRA.NS", "GNFC.NS", "GODREJPROP.NS", "GRANULES.NS",
    "GSPL.NS", "GUJGASLTD.NS", "HAPPSTMNDS.NS", "HINDPETRO.NS",
    "HONAUT.NS", "IDFCFIRSTB.NS", "IEX.NS", "IIFL.NS",
    "INDIACEM.NS", "INDIAMART.NS", "INDIANB.NS", "INDUSTOWER.NS",
    "IREDA.NS", "IPCALAB.NS", "JKCEMENT.NS", "JUBLFOOD.NS",
    "KALYANKJIL.NS", "KANSAINER.NS", "KEI.NS", "KPITTECH.NS",
    "L&TFH.NS", "LATENTVIEW.NS", "LAURUSLABS.NS", "LICHSGFIN.NS",
    "LUPIN.NS", "M&MFIN.NS", "MANAPPURAM.NS", "MFSL.NS",
    "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS",
    "MUTHOOTFIN.NS", "NAM-INDIA.NS", "NATIONALUM.NS", "NAUKRI.NS",
    "NAVINFLUOR.NS", "NMDC.NS", "OBEROIRLTY.NS", "OFSS.NS",
    "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PIIND.NS",
    "POLYCAB.NS", "PRESTIGE.NS", "PVRINOX.NS", "RAMCOCEM.NS",
    "RBLBANK.NS", "SAIL.NS", "SCHAEFFLER.NS", "SHREECEM.NS",
    "SONACOMS.NS", "STARHEALTH.NS", "SUNDARMFIN.NS", "SUNDRMFAST.NS",
    "SUPREMEIND.NS", "SYNGENE.NS", "TATACHEM.NS", "TATACOMM.NS",
    "THERMAX.NS", "TIMKEN.NS", "TORNTPOWER.NS", "TRENT.NS",
    "TRIDENT.NS", "UBL.NS", "UNIONBANK.NS", "UPL.NS",
    "VOLTAS.NS", "WHIRLPOOL.NS", "YESBANK.NS", "ZEEL.NS",
]

# --- NIFTY SMALLCAP & OTHER POPULAR ---
NIFTY_SMALLCAP_OTHER = [
    "3MINDIA.NS", "AAVAS.NS", "AFFLE.NS", "AIAENG.NS",
    "AMARAJABAT.NS", "ANGELONE.NS", "APARINDS.NS", "ASHOKLEY.NS",
    "ATUL.NS", "AVANTIFEED.NS", "AXISBANK.NS", "BANARISUG.NS",
    "BASF.NS", "BAYERCROP.NS", "BBTC.NS", "BDL.NS",
    "BEML.NS", "BHARATFORG.NS", "BIKAJI.NS", "BLUESTARCO.NS",
    "BSE.NS", "CAMPUS.NS", "CANFINHOME.NS", "CARBORUNIV.NS",
    "CASTROLIND.NS", "CDSL.NS", "CENTURYTEX.NS", "CERA.NS",
    "CHALET.NS", "CLEAN.NS", "COCHINSHIP.NS", "CYIENT.NS",
    "DATAPATTNS.NS", "DCMSHRIRAM.NS", "DELHIVERY.NS", "DHANI.NS",
    "ECLERX.NS", "EDELWEISS.NS", "EIHOTEL.NS", "ELGIEQUIP.NS",
    "EQUITASBNK.NS", "FACT.NS", "FINCABLES.NS", "FINEORG.NS",
    "FSL.NS", "GICRE.NS", "GILLETTE.NS", "GLAXO.NS",
    "GOCOLORS.NS", "GRINDWELL.NS", "GRSE.NS", "GSFC.NS",
    "GSHIP.NS", "Gujarat.NS", "HEG.NS", "HFCL.NS",
    "HINDCOPPER.NS", "HINDZINC.NS", "HOMEFIRST.NS", "HUDCO.NS",
    "IDBI.NS", "IDEA.NS", "IFBIND.NS", "IIFLWAM.NS",
    "INDHOTEL.NS", "INTELLECT.NS", "IONEXCHANG.NS", "ISEC.NS",
    "ITI.NS", "JBCHEPHARM.NS", "JKLAKSHMI.NS", "JMFINANCIL.NS",
    "JSL.NS", "JSWINFRA.NS", "JTEKTINDIA.NS", "KAJARIACER.NS",
    "KALPATPOWR.NS", "KEC.NS", "KNRCON.NS", "KPRMILL.NS",
    "KSB.NS", "LAXMIMACH.NS", "LINDEINDIA.NS", "LLOYDSME.NS",
    "LTF.NS", "LTTS.NS", "MAHABANK.NS", "MAHINDCIE.NS",
    "MAHSEAMLES.NS", "MASTEK.NS", "MAXHEALTH.NS", "MAZAGON.NS",
    "MCX.NS", "METROPOLIS.NS", "MMTC.NS", "MOIL.NS",
    "MRPL.NS", "NATCOPHARM.NS", "NIACL.NS", "NLCINDIA.NS",
    "OLECTRA.NS", "PGHH.NS", "PHOENIXLTD.NS", "POLYMED.NS",
    "POONAWALLA.NS", "PRSMJOHNSN.NS", "QUESS.NS", "RADICO.NS",
    "RAJESHEXPO.NS", "RALLIS.NS", "RITES.NS", "ROUTE.NS",
    "RVNL.NS", "SAPPHIRE.NS", "SANOFI.NS", "SCHNEIDER.NS",
    "SJVN.NS", "SKFINDIA.NS", "SOBHA.NS", "SOLARINDS.NS",
    "SPARC.NS", "STAR.NS", "SUNTV.NS", "SUPRAJIT.NS",
    "SUVENPHAR.NS", "SUZLON.NS", "SYMPHONY.NS", "TANLA.NS",
    "TATAINVEST.NS", "TCIEXP.NS", "TEJASNET.NS", "TIINDIA.NS",
    "TINPLATE.NS", "TRIVENI.NS", "UCOBANK.NS", "UJJIVAN.NS",
    "UTIAMC.NS", "VAIBHAVGBL.NS", "VAKRANGEE.NS", "VGUARD.NS",
    "VINATIORGA.NS", "VMART.NS", "VSTIND.NS", "WELCORP.NS",
    "WELSPUNLIV.NS", "WOCKPHARMA.NS", "ZENSARTECH.NS",
]

# NSE official equity symbol master (broad coverage beyond curated lists)
_NSE_MASTER_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
_CACHE_PATH = Path(__file__).parent / "cache" / "EQUITY_L.csv"
_CACHE_TTL_DAYS = 7


def _normalize_nse_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if not s:
        return ""
    if s.endswith(".NS"):
        return s
    return f"{s}.NS"


def _parse_master_symbols(csv_text: str) -> list[str]:
    symbols: set[str] = set()
    reader = csv.DictReader(StringIO(csv_text))
    for row in reader:
        sym = _normalize_nse_symbol(row.get("SYMBOL", ""))
        if sym:
            symbols.add(sym)
    return sorted(symbols)


def _load_master_from_cache() -> list[str]:
    if not _CACHE_PATH.exists():
        return []
    try:
        mtime = datetime.fromtimestamp(_CACHE_PATH.stat().st_mtime)
        if datetime.now() - mtime > timedelta(days=_CACHE_TTL_DAYS):
            return []
        text = _CACHE_PATH.read_text(encoding="utf-8")
        return _parse_master_symbols(text)
    except Exception:
        return []


def _fetch_master_symbols() -> list[str]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/csv,*/*",
            "Referer": "https://www.nseindia.com/",
        }
        resp = requests.get(_NSE_MASTER_URL, headers=headers, timeout=2)
        resp.raise_for_status()
        text = resp.text

        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(text, encoding="utf-8")

        return _parse_master_symbols(text)
    except Exception:
        return []


def _load_all_nse_symbols() -> list[str]:
    cached = _load_master_from_cache()
    if cached:
        return cached
    return _fetch_master_symbols()


# Symbols that have changed on the exchange/data providers. Legacy names remain
# accepted by the UI but are canonicalized before retrieval and ranking.
TICKER_ALIASES = {
    "ZOMATO.NS": "ETERNAL.NS",
}

# --- Combined & deduplicated list ---
ALL_NSE_TICKERS = sorted(
    set(
        NIFTY_50
        + NIFTY_NEXT_50
        + NIFTY_MIDCAP
        + NIFTY_SMALLCAP_OTHER
        + _load_all_nse_symbols()
    ) - set(TICKER_ALIASES)
)

# Quick lookup set
NSE_TICKER_SET = set(ALL_NSE_TICKERS)
