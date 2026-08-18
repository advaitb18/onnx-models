#property strict
#property version   "2.00"
#property description "Generic independent multi-strategy portfolio executor"

#include <Trade/Trade.mqh>

CTrade trade;


//==============================================================
// INPUTS
//==============================================================

input string InpPortfolioRoot =
   "xau_signals\\portfolio";

input string InpQueueFile =
   "xau_signals\\portfolio\\queue.txt";

input string InpStrategyContractsFile =
   "xau_signals\\portfolio\\strategy_contracts.csv";

input double InpLots = 0.01;

input bool InpExecutionEnabled = false;

input bool InpOnePositionPerStrategy = true;

input bool InpAllowBuy = true;
input bool InpAllowSell = true;

input int InpTimerSeconds = 2;

input int InpDeviationPoints = 30;


//==============================================================
// SIGNAL CONTRACT
//==============================================================

struct PortfolioSignal
{
   int      schema_version;

   string   signal_id;
   string   strategy_id;
   string   model_id;

   string   symbol;
   string   timeframe;
   string   decision;
   string   entry_type;

   datetime created_at;
   datetime expires_at;

   string   signal_bar_timestamp;

   double   model_score;
   double   rank_pct;
   double   decision_threshold;

   double   atr;
   double   tp_atr;
   double   sl_atr;

   int      max_hold_bars;
   long     magic;
};


//==============================================================
// LOGGING
//==============================================================

void Log(string text)
{
   Print(
      "[PORTFOLIO_V02] ",
      text
   );
}


//==============================================================
// STRING HELPERS
//==============================================================

string Trim(string value)
{
   StringTrimLeft(value);
   StringTrimRight(value);
   return value;
}


bool SplitKeyValue(
   const string line,
   string &key,
   string &value
)
{
   int pos = StringFind(line, "=");

   if(pos < 1)
      return false;

   key = Trim(
      StringSubstr(line, 0, pos)
   );

   value = Trim(
      StringSubstr(line, pos + 1)
   );

   return true;
}


//==============================================================
// SIGNAL READER
//==============================================================

bool ReadSignal(
   const string filename,
   PortfolioSignal &s
)
{
   string path =
      InpPortfolioRoot
      + "\\inbox\\"
      + filename;

   ResetLastError();

   int h = FileOpen(
      path,
      FILE_READ
      | FILE_TXT
      | FILE_COMMON
      | FILE_ANSI
   );

   if(h == INVALID_HANDLE)
   {
      Log(
         "READ_FAILED file="
         + filename
         + " err="
         + IntegerToString(GetLastError())
      );

      return false;
   }

   ZeroMemory(s);

   while(!FileIsEnding(h))
   {
      string line = FileReadString(h);

      if(StringLen(line) == 0)
         continue;

      string key;
      string value;

      if(!SplitKeyValue(
         line,
         key,
         value
      ))
         continue;


      if(key == "schema_version")
         s.schema_version =
            (int)StringToInteger(value);

      else if(key == "signal_id")
         s.signal_id = value;

      else if(key == "strategy_id")
         s.strategy_id = value;

      else if(key == "model_id")
         s.model_id = value;

      else if(key == "symbol")
         s.symbol = value;

      else if(key == "timeframe")
         s.timeframe = value;

      else if(key == "decision")
         s.decision = value;

      else if(key == "entry_type")
         s.entry_type = value;

      else if(key == "created_at_epoch")
         s.created_at =
            (datetime)StringToInteger(value);

      else if(key == "expires_at_epoch")
         s.expires_at =
            (datetime)StringToInteger(value);

      else if(key == "signal_bar_timestamp")
         s.signal_bar_timestamp = value;

      else if(key == "model_score")
         s.model_score =
            StringToDouble(value);

      else if(key == "rank_pct")
      {
         if(StringLen(value) > 0)
            s.rank_pct =
               StringToDouble(value);
      }

      else if(key == "decision_threshold")
         s.decision_threshold =
            StringToDouble(value);

      else if(key == "atr")
         s.atr =
            StringToDouble(value);

      else if(key == "tp_atr")
         s.tp_atr =
            StringToDouble(value);

      else if(key == "sl_atr")
         s.sl_atr =
            StringToDouble(value);

      else if(key == "max_hold_bars")
         s.max_hold_bars =
            (int)StringToInteger(value);

      else if(key == "magic")
         s.magic =
            (long)StringToInteger(value);
   }

   FileClose(h);

   if(s.schema_version != 2)
   {
      Log(
         "INVALID_SCHEMA file="
         + filename
      );

      return false;
   }

   if(StringLen(s.signal_id) == 0)
      return false;

   if(StringLen(s.strategy_id) == 0)
      return false;

   if(s.magic <= 0)
      return false;

   return true;
}


//==============================================================
// POSITION LOOKUP
//==============================================================

bool HasStrategyPosition(
   const long magic
)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket =
         PositionGetTicket(i);

      if(ticket == 0)
         continue;

      if(!PositionSelectByTicket(ticket))
         continue;

      long position_magic =
         PositionGetInteger(
            POSITION_MAGIC
         );

      string symbol =
         PositionGetString(
            POSITION_SYMBOL
         );

      if(
         position_magic == magic
         &&
         symbol == _Symbol
      )
         return true;
   }

   return false;
}


//==============================================================
// TIMEFRAME CONVERSION
//==============================================================

ENUM_TIMEFRAMES ParseTimeframe(
   const string tf
)
{
   if(tf == "M1")
      return PERIOD_M1;

   if(tf == "M5")
      return PERIOD_M5;

   if(tf == "M15")
      return PERIOD_M15;

   if(tf == "M30")
      return PERIOD_M30;

   if(tf == "H1")
      return PERIOD_H1;

   if(tf == "H4")
      return PERIOD_H4;

   if(tf == "D1")
      return PERIOD_D1;

   return PERIOD_CURRENT;
}


//==============================================================
// FILE FINALIZATION
//==============================================================

bool MoveSignal(
   const string filename,
   const string destination
)
{
   string src =
      InpPortfolioRoot
      + "\\inbox\\"
      + filename;

   string dst =
      InpPortfolioRoot
      + "\\"
      + destination
      + "\\"
      + filename;

   ResetLastError();

   if(FileMove(
      src,
      FILE_COMMON,
      dst,
      FILE_COMMON
   ))
      return true;

   Log(
      "FILE_MOVE_FAILED file="
      + filename
      + " destination="
      + destination
      + " err="
      + IntegerToString(GetLastError())
   );

   return false;
}


//==============================================================
// SIGNAL EXECUTION
//==============================================================

void ProcessSignal(
   const string filename
)
{
   PortfolioSignal s;

	string inbox_path =
	   InpPortfolioRoot
	   + "\\inbox\\"
	   + filename;

	// A stale queue entry is normal after another cycle
	// has already consumed/moved the signal.
	if(!FileIsExist(
	   inbox_path,
	   FILE_COMMON
	))
	{
	   return;
	}

	if(!ReadSignal(
	   filename,
	   s
	))
	{
	   Log(
	      "REJECT_UNREADABLE_SIGNAL file="
	      + filename
	   );

	   MoveSignal(
	      filename,
	      "rejected"
	   );

	   return;
	}


   // ---------------------------------------------------------
   // Expiry
   // ---------------------------------------------------------

   if(
      s.expires_at > 0
      &&
      TimeGMT() > s.expires_at
   )
   {
      Log(
         "EXPIRED "
         + s.strategy_id
         + " signal="
         + s.signal_id
      );

      MoveSignal(
         filename,
         "expired"
      );

      return;
   }


   // ---------------------------------------------------------
   // Contract checks
   // ---------------------------------------------------------

   if(s.entry_type != "MARKET")
   {
      Log(
         "REJECT_NON_MARKET "
         + s.strategy_id
      );

      MoveSignal(
         filename,
         "rejected"
      );

      return;
   }


   bool buy =
      s.decision == "BUY";

   bool sell =
      s.decision == "SELL";


   if(!buy && !sell)
   {
      Log(
         "REJECT_DIRECTION "
         + s.strategy_id
      );

      MoveSignal(
         filename,
         "rejected"
      );

      return;
   }


   if(buy && !InpAllowBuy)
   {
      Log(
         "BUY_DISABLED "
         + s.strategy_id
      );

      return;
   }


   if(sell && !InpAllowSell)
   {
      Log(
         "SELL_DISABLED "
         + s.strategy_id
      );

      return;
   }


   if(
      s.atr <= 0
      ||
      s.tp_atr <= 0
      ||
      s.sl_atr <= 0
   )
   {
      Log(
         "REJECT_INVALID_RISK "
         + s.strategy_id
      );

      MoveSignal(
         filename,
         "rejected"
      );

      return;
   }


   // ---------------------------------------------------------
   // One position PER STRATEGY, not globally.
   // ---------------------------------------------------------

   if(
      InpOnePositionPerStrategy
      &&
      HasStrategyPosition(s.magic)
   )
   {
      Log(
         "POSITION_EXISTS "
         + s.strategy_id
         + " magic="
         + IntegerToString((int)s.magic)
      );

      MoveSignal(
         filename,
         "processed"
      );

      return;
   }


   // ---------------------------------------------------------
   // Safety gate
   // ---------------------------------------------------------

   if(!InpExecutionEnabled)
   {
      Log(
         "DRY_RUN QUALIFIED "
         + s.strategy_id
         + " "
         + s.decision
         + " signal="
         + s.signal_id
      );

      // Do NOT consume during dry-run.
      return;
   }


   // ---------------------------------------------------------
   // Current broker price
   // ---------------------------------------------------------

   MqlTick tick;

   if(!SymbolInfoTick(
      _Symbol,
      tick
   ))
   {
      Log(
         "NO_TICK "
         + s.strategy_id
      );

      return;
   }


   double entry =
      buy
      ? tick.ask
      : tick.bid;


   double sl_distance =
      s.atr * s.sl_atr;

   double tp_distance =
      s.atr * s.tp_atr;


   double sl;
   double tp;


   if(buy)
   {
      sl = NormalizeDouble(
         entry - sl_distance,
         _Digits
      );

      tp = NormalizeDouble(
         entry + tp_distance,
         _Digits
      );
   }
   else
   {
      sl = NormalizeDouble(
         entry + sl_distance,
         _Digits
      );

      tp = NormalizeDouble(
         entry - tp_distance,
         _Digits
      );
   }


   // ---------------------------------------------------------
   // Execute
   // ---------------------------------------------------------

   trade.SetExpertMagicNumber(
      s.magic
   );

   trade.SetDeviationInPoints(
      InpDeviationPoints
   );


   string comment =
      s.strategy_id
      + "|"
      + s.signal_id;


   bool ok = false;


   if(buy)
   {
      ok = trade.Buy(
         InpLots,
         _Symbol,
         0.0,
         sl,
         tp,
         comment
      );
   }
   else
   {
      ok = trade.Sell(
         InpLots,
         _Symbol,
         0.0,
         sl,
         tp,
         comment
      );
   }


   if(ok)
   {
      Log(
         "EXECUTED "
         + s.strategy_id
         + " "
         + s.decision
         + " magic="
         + IntegerToString((int)s.magic)
         + " lots="
         + DoubleToString(
              InpLots,
              2
           )
         + " entry="
         + DoubleToString(
              entry,
              _Digits
           )
         + " sl="
         + DoubleToString(
              sl,
              _Digits
           )
         + " tp="
         + DoubleToString(
              tp,
              _Digits
           )
      );

      MoveSignal(
         filename,
         "processed"
      );
   }
   else
   {
      Log(
         "ORDER_FAILED "
         + s.strategy_id
         + " retcode="
         + IntegerToString(
              (int)trade.ResultRetcode()
           )
         + " "
         + trade.ResultRetcodeDescription()
      );

      // Leave in inbox so temporary broker errors
      // can be retried before signal expiry.
   }
}


//==============================================================
// QUEUE PROCESSOR
//==============================================================

void ProcessQueue()
{
   ResetLastError();

   int h = FileOpen(
      InpQueueFile,
      FILE_READ
      | FILE_TXT
      | FILE_COMMON
      | FILE_ANSI
      | FILE_SHARE_READ
   );

   if(h == INVALID_HANDLE)
      return;


   while(!FileIsEnding(h))
   {
      string filename =
         Trim(
            FileReadString(h)
         );

      if(StringLen(filename) == 0)
         continue;

      ProcessSignal(filename);
   }


   FileClose(h);
}

//==============================================================
// GENERIC STRATEGY CONTRACT LOOKUP
//==============================================================

bool LookupStrategyContract(
   const long magic,
   ENUM_TIMEFRAMES &tf,
   int &max_bars,
   string &strategy_id
)
{
   tf = PERIOD_CURRENT;
   max_bars = 0;
   strategy_id = "";

   ResetLastError();

   int h = FileOpen(
      InpStrategyContractsFile,
      FILE_READ
      | FILE_CSV
      | FILE_COMMON
      | FILE_ANSI,
      ','
   );

   if(h == INVALID_HANDLE)
   {
      Log(
         "CONTRACT_FILE_OPEN_FAILED err="
         + IntegerToString(GetLastError())
      );

      return false;
   }

   while(!FileIsEnding(h))
   {
      string magic_text =
         FileReadString(h);

      string row_strategy_id =
         FileReadString(h);

      string timeframe_text =
         FileReadString(h);

      string max_bars_text =
         FileReadString(h);


      // Header
      if(magic_text == "magic")
         continue;


      long row_magic =
         (long)StringToInteger(
            magic_text
         );


      if(row_magic != magic)
         continue;


      ENUM_TIMEFRAMES parsed_tf =
         ParseTimeframe(
            timeframe_text
         );


      int parsed_max_bars =
         (int)StringToInteger(
            max_bars_text
         );


      if(
         parsed_tf == PERIOD_CURRENT
         ||
         parsed_max_bars <= 0
      )
      {
         FileClose(h);

         Log(
            "INVALID_STRATEGY_CONTRACT magic="
            + IntegerToString((int)magic)
         );

         return false;
      }


      tf = parsed_tf;
      max_bars = parsed_max_bars;
      strategy_id = row_strategy_id;

      FileClose(h);

      return true;
   }


   FileClose(h);

   return false;
}

//==============================================================
// MAX-HOLD POSITION MANAGEMENT
//==============================================================

void ManagePositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket =
         PositionGetTicket(i);

      if(ticket == 0)
         continue;

      if(!PositionSelectByTicket(ticket))
         continue;

      if(
         PositionGetString(POSITION_SYMBOL)
         != _Symbol
      )
         continue;


      string comment =
         PositionGetString(
            POSITION_COMMENT
         );

      // Only manage positions created by this portfolio.
      if(StringFind(comment, "|") < 1)
         continue;


      datetime opened =
         (datetime)PositionGetInteger(
            POSITION_TIME
         );

      long magic =
         PositionGetInteger(
            POSITION_MAGIC
         );


      // Current two strategies.
      // Future version will persist max_hold_bars by magic.
      int max_bars = 0;
      ENUM_TIMEFRAMES tf =
         PERIOD_CURRENT;


	string strategy_id = "";


      if(!LookupStrategyContract(
	 magic,
	 tf,
	 max_bars,
	 strategy_id
      ))
      {
	   // Position does not belong to an enabled
	   // strategy known by this portfolio registry.
	 continue;
      }

      int seconds =
         PeriodSeconds(tf);

      if(
         seconds <= 0
         ||
         max_bars <= 0
      )
         continue;


      datetime deadline =
         opened
         + (
            seconds
            * max_bars
           );


      if(TimeCurrent() < deadline)
         continue;


      trade.SetExpertMagicNumber(
         magic
      );


      if(trade.PositionClose(ticket))
      {
         Log(
            "MAX_HOLD_CLOSE ticket="
            + IntegerToString((int)ticket)
            + " magic="
            + IntegerToString((int)magic)
         );
      }
      else
      {
         Log(
            "MAX_HOLD_CLOSE_FAILED ticket="
            + IntegerToString((int)ticket)
            + " "
            + trade.ResultRetcodeDescription()
         );
      }
   }
}


//==============================================================
// LIFECYCLE
//==============================================================

int OnInit()
{
   EventSetTimer(
      MathMax(
         1,
         InpTimerSeconds
      )
   );

   Log(
      "INIT"
      + StringFormat(
           " execution=%s symbol=%s lots=%.2f",
           InpExecutionEnabled
           ? "ENABLED"
           : "DRY_RUN",
           _Symbol,
           InpLots
        )
   );

   return INIT_SUCCEEDED;
}


void OnDeinit(
   const int reason
)
{
   EventKillTimer();
}


void OnTimer()
{
   ProcessQueue();

   if(InpExecutionEnabled)
      ManagePositions();
}


void OnTick()
{
}
