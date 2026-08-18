//+------------------------------------------------------------------+
//| XAUUSD_MTF_BAR_EXPORTER_P8A1.mq5                                 |
//| P8A1 MT5 multi-timeframe OHLCV exporter                          |
//|                                                                  |
//| Purpose:                                                         |
//| - Export closed M5/M15/H1/H4/D1 bars to MT5 Common Files         |
//| - Python P8/V05H live feature bridge consumes these CSVs          |
//| - No signal writing                                              |
//| - No trading                                                     |
//| - No CTrade                                                      |
//+------------------------------------------------------------------+
#property strict
#property version   "8.20"
#property description "P8A1 XAUUSD MTF bar exporter for Python live feature bridge. M5 enabled."

input string InpSymbol                 = "";       // blank = use chart symbol
input string InpOutputDirCommon        = "xau_signals";
input int    InpBarsM5                 = 2000;
input int    InpBarsM15                = 1500;
input int    InpBarsH1                 = 750;
input int    InpBarsH4                 = 500;
input int    InpBarsD1                 = 300;
input int    InpTimerSeconds           = 2;
input bool   InpExportOnInit           = true;
input bool   InpExportOnlyClosedBars   = true;
input bool   InpLogEveryExport         = true;
input bool   InpWriteHeartbeat         = true;

datetime g_last_closed_m15_bar_time = 0;
datetime g_last_export_time         = 0;
int      g_export_count             = 0;

string EffectiveSymbol()
{
   string s = InpSymbol;
   StringTrimLeft(s);
   StringTrimRight(s);

   if(s == "")
      return _Symbol;

   return s;
}

string TimeframeToLabel(const ENUM_TIMEFRAMES tf)
{
   if(tf == PERIOD_M5)  return "m5";
   if(tf == PERIOD_M15) return "m15";
   if(tf == PERIOD_H1)  return "h1";
   if(tf == PERIOD_H4)  return "h4";
   if(tf == PERIOD_D1)  return "d1";
   return "unknown";
}

string TimeframeToText(const ENUM_TIMEFRAMES tf)
{
   if(tf == PERIOD_M5)  return "M5";
   if(tf == PERIOD_M15) return "M15";
   if(tf == PERIOD_H1)  return "H1";
   if(tf == PERIOD_H4)  return "H4";
   if(tf == PERIOD_D1)  return "D1";
   return "UNKNOWN";
}

string CsvPathForTf(const ENUM_TIMEFRAMES tf)
{
   return InpOutputDirCommon + "\\xauusd_" + TimeframeToLabel(tf) + "_bars.csv";
}

string HeartbeatPath()
{
   return InpOutputDirCommon + "\\xauusd_bar_export_heartbeat.txt";
}

bool EnsureCommonDir()
{
   ResetLastError();
   bool ok = FolderCreate(InpOutputDirCommon, FILE_COMMON);
   int folder_err = GetLastError();

   if(ok)
      return true;

   string test_path = InpOutputDirCommon + "\\p8a1_write_test.tmp";

   ResetLastError();
   int h = FileOpen(test_path, FILE_WRITE | FILE_TXT | FILE_COMMON | FILE_ANSI);

   if(h == INVALID_HANDLE)
   {
      Print("P8A1 ERROR cannot create/common-dir or test file. dir=", InpOutputDirCommon,
            " err=", GetLastError(), " folder_err=", folder_err);
      return false;
   }

   FileWriteString(h, "ok\n");
   FileClose(h);
   FileDelete(test_path, FILE_COMMON);

   return true;
}

datetime CurrentClosedM15BarTime()
{
   int shift = InpExportOnlyClosedBars ? 1 : 0;
   return iTime(EffectiveSymbol(), PERIOD_M15, shift);
}

int ExportBars(const ENUM_TIMEFRAMES tf, const int bars_requested, string &status)
{
   status = "";

   int shift_start = InpExportOnlyClosedBars ? 1 : 0;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   ResetLastError();

   string sym = EffectiveSymbol();
   SymbolSelect(sym, true);

   int copied = CopyRates(sym, tf, shift_start, bars_requested, rates);

   if(copied <= 0)
   {
      status = "COPYRATES_FAILED tf=" + TimeframeToText(tf) + " err=" + IntegerToString(GetLastError());
      return 0;
   }

   string path = CsvPathForTf(tf);

   ResetLastError();
   int h = FileOpen(path, FILE_WRITE | FILE_TXT | FILE_COMMON | FILE_ANSI);

   if(h == INVALID_HANDLE)
   {
      status = "FILE_OPEN_FAILED path=" + path + " err=" + IntegerToString(GetLastError());
      return 0;
   }

   FileWriteString(h, "timestamp_utc,time_epoch,open,high,low,close,volume,tick_volume,spread,real_volume,timeframe,symbol,is_closed_bar\n");

   // Current broker/server offset from GMT.
  // Example: broker=10:00, GMT=07:00 -> offset=+10800 seconds.
   long broker_gmt_offset = (long)(TimeCurrent() - TimeGMT());

   for(int i = copied - 1; i >= 0; i--)
   {
   // CopyRates bar time follows the broker/server chart clock in this feed.
   // Convert it to true UTC before exporting to Python.
	datetime broker_bar_time = rates[i].time;
	datetime utc_bar_time =
	(datetime)((long)broker_bar_time - broker_gmt_offset);
	string timestamp = TimeToString(
	utc_bar_time,
	TIME_DATE | TIME_MINUTES | TIME_SECONDS
   );

      long tick_volume = (long)rates[i].tick_volume;
      long real_volume = (long)rates[i].real_volume;
      long volume_for_python = (real_volume > 0) ? real_volume : tick_volume;

      string line = "";
      line += timestamp + ",";
      line += IntegerToString((long)utc_bar_time) + ",";
      line += DoubleToString(rates[i].open, _Digits) + ",";
      line += DoubleToString(rates[i].high, _Digits) + ",";
      line += DoubleToString(rates[i].low, _Digits) + ",";
      line += DoubleToString(rates[i].close, _Digits) + ",";
      line += IntegerToString(volume_for_python) + ",";
      line += IntegerToString(tick_volume) + ",";
      line += IntegerToString((int)rates[i].spread) + ",";
      line += IntegerToString(real_volume) + ",";
      line += TimeframeToText(tf) + ",";
      line += EffectiveSymbol() + ",";
      line += (InpExportOnlyClosedBars ? "true" : "false");
      line += "\n";

      FileWriteString(h, line);
   }

   FileClose(h);

   datetime oldest = rates[copied - 1].time;
   datetime newest = rates[0].time;

   status = "OK tf=" + TimeframeToText(tf)
            + " requested=" + IntegerToString(bars_requested)
            + " copied=" + IntegerToString(copied)
            + " oldest=" + TimeToString(oldest, TIME_DATE | TIME_MINUTES)
            + " newest=" + TimeToString(newest, TIME_DATE | TIME_MINUTES)
            + " path=" + path;

   return copied;
}

bool WriteHeartbeat(
   const string trigger,
   const int m5_count,
   const int m15_count,
   const int h1_count,
   const int h4_count,
   const int d1_count,
   const string m5_status,
   const string m15_status,
   const string h1_status,
   const string h4_status,
   const string d1_status
)
{
   if(!InpWriteHeartbeat)
      return true;

   string path = HeartbeatPath();

   ResetLastError();
   int h = FileOpen(path, FILE_WRITE | FILE_TXT | FILE_COMMON | FILE_ANSI);

   if(h == INVALID_HANDLE)
   {
      Print("P8A1 HEARTBEAT_WRITE_FAILED path=", path, " err=", GetLastError());
      return false;
   }

   datetime closed_m15 = CurrentClosedM15BarTime();

   FileWriteString(h, "phase=P8A1\n");
   FileWriteString(h, "engine_version=P8A1_MTF_BAR_EXPORTER_M5_ENABLED\n");
   FileWriteString(h, "symbol=" + EffectiveSymbol() + "\n");
   FileWriteString(h, "chart_symbol=" + _Symbol + "\n");
   FileWriteString(h, "input_symbol=" + InpSymbol + "\n");
   FileWriteString(h, "mode=BAR_EXPORT_ONLY\n");
   FileWriteString(h, "trading_enabled=false\n");
   FileWriteString(h, "signals_written=false\n");
   FileWriteString(h, "export_trigger=" + trigger + "\n");
   FileWriteString(h, "last_export_time_broker=" + TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES | TIME_SECONDS) + "\n");
   FileWriteString(h, "last_closed_m15_bar_time=" + TimeToString(closed_m15, TIME_DATE | TIME_MINUTES | TIME_SECONDS) + "\n");
   FileWriteString(h, "last_closed_m15_bar_epoch=" + IntegerToString((long)closed_m15) + "\n");
   FileWriteString(h, "export_count=" + IntegerToString(g_export_count) + "\n");

   FileWriteString(h, "m5_bars_exported=" + IntegerToString(m5_count) + "\n");
   FileWriteString(h, "m15_bars_exported=" + IntegerToString(m15_count) + "\n");
   FileWriteString(h, "h1_bars_exported=" + IntegerToString(h1_count) + "\n");
   FileWriteString(h, "h4_bars_exported=" + IntegerToString(h4_count) + "\n");
   FileWriteString(h, "d1_bars_exported=" + IntegerToString(d1_count) + "\n");

   FileWriteString(h, "m5_csv=" + CsvPathForTf(PERIOD_M5) + "\n");
   FileWriteString(h, "m15_csv=" + CsvPathForTf(PERIOD_M15) + "\n");
   FileWriteString(h, "h1_csv=" + CsvPathForTf(PERIOD_H1) + "\n");
   FileWriteString(h, "h4_csv=" + CsvPathForTf(PERIOD_H4) + "\n");
   FileWriteString(h, "d1_csv=" + CsvPathForTf(PERIOD_D1) + "\n");

   FileWriteString(h, "m5_status=" + m5_status + "\n");
   FileWriteString(h, "m15_status=" + m15_status + "\n");
   FileWriteString(h, "h1_status=" + h1_status + "\n");
   FileWriteString(h, "h4_status=" + h4_status + "\n");
   FileWriteString(h, "d1_status=" + d1_status + "\n");

   FileClose(h);

   return true;
}

bool ExportAll(const string trigger)
{
   if(!EnsureCommonDir())
      return false;

   string s_m5  = "";
   string s_m15 = "";
   string s_h1  = "";
   string s_h4  = "";
   string s_d1  = "";

   int c_m5  = ExportBars(PERIOD_M5,  InpBarsM5,  s_m5);
   int c_m15 = ExportBars(PERIOD_M15, InpBarsM15, s_m15);
   int c_h1  = ExportBars(PERIOD_H1,  InpBarsH1,  s_h1);
   int c_h4  = ExportBars(PERIOD_H4,  InpBarsH4,  s_h4);
   int c_d1  = ExportBars(PERIOD_D1,  InpBarsD1,  s_d1);

   g_export_count++;
   g_last_export_time = TimeCurrent();

   bool heartbeat_ok = WriteHeartbeat(
      trigger,
      c_m5,
      c_m15,
      c_h1,
      c_h4,
      c_d1,
      s_m5,
      s_m15,
      s_h1,
      s_h4,
      s_d1
   );

   if(InpLogEveryExport)
   {
      Print("P8A1 EXPORT trigger=", trigger,
            " m5=", c_m5,
            " m15=", c_m15,
            " h1=", c_h1,
            " h4=", c_h4,
            " d1=", c_d1,
            " heartbeat=", heartbeat_ok);

      Print("P8A1 STATUS ", s_m5);
      Print("P8A1 STATUS ", s_m15);
      Print("P8A1 STATUS ", s_h1);
      Print("P8A1 STATUS ", s_h4);
      Print("P8A1 STATUS ", s_d1);
   }

   return (c_m5 > 0 && c_m15 > 0 && c_h1 > 0 && c_h4 > 0 && c_d1 > 0 && heartbeat_ok);
}

void CheckNewM15BarAndExport()
{
   datetime closed_m15 = CurrentClosedM15BarTime();

   if(closed_m15 <= 0)
   {
      Print("P8A1 WAITING_FOR_M15_HISTORY symbol=", EffectiveSymbol(), " chart_symbol=", _Symbol);
      return;
   }

   if(g_last_closed_m15_bar_time == 0)
   {
      g_last_closed_m15_bar_time = closed_m15;

      if(InpExportOnInit)
         ExportAll("INIT_OR_FIRST_TIMER");

      return;
   }

   if(closed_m15 != g_last_closed_m15_bar_time)
   {
      g_last_closed_m15_bar_time = closed_m15;
      ExportAll("NEW_M15_BAR");
      return;
   }
}

int OnInit()
{
   SymbolSelect(EffectiveSymbol(), true);

   Print("P8A1 INIT symbol=", EffectiveSymbol(),
         " chart_symbol=", _Symbol,
         " input_symbol=", InpSymbol,
         " output_dir_common=", InpOutputDirCommon,
         " bars_m5=", InpBarsM5,
         " bars_m15=", InpBarsM15,
         " bars_h1=", InpBarsH1,
         " bars_h4=", InpBarsH4,
         " bars_d1=", InpBarsD1,
         " export_only_closed=", InpExportOnlyClosedBars,
         " trading=false signals=false");

   EventSetTimer(InpTimerSeconds);

   if(InpExportOnInit)
   {
      g_last_closed_m15_bar_time = CurrentClosedM15BarTime();
      ExportAll("ON_INIT");
   }

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("P8A1 DEINIT reason=", reason, " exports=", g_export_count);
}

void OnTimer()
{
   CheckNewM15BarAndExport();
}

void OnTick()
{
   // Timer-driven by design.
}
//+------------------------------------------------------------------+
