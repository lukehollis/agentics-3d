// HugeGregorianDateTime
// Use at your own risk.
// Contact for commercial license. charlybeck@gmx.net
// Version 202211022248

using System;
using System.Linq;
using UnityEngine;

namespace CbAutorenTool.Tools
{

    /// <summary>
    /// Greogorian date time using an UInt64 and resolution of seconds,
    /// allowing to store and calculate very huge timespan from
    /// year ~290.000.000.000 BC (before christ) to
    /// year ~290.000.000.000 AC (After christ) 
    /// </summary>
    public sealed class CHugeDateTimeOffset : CHugeDateTimeBase
    {
        public CHugeDateTimeOffset()
        {

        }
        public CHugeDateTimeOffset(int y, int mm, int d, int h, int m, int s)
        {
            this.Years = y;
            this.Months = mm;
            this.Days = d;
            this.Hours = h;
            this.Minutes = m;
            this.Seconds = s;
        }
        public CHugeDateTimeOffset(int[] i)
            :
            this
            (
                GetInt(i, 0),
                GetInt(i, 1),
                GetInt(i, 2),
                GetInt(i, 3),
                GetInt(i, 4),
                GetInt(i, 5)
            )
        {
        }
        public int Years { get; private set; }
        public int Months { get; private set; }
        public int Days { get; private set; }
        public int Hours { get; private set; }
        public int Minutes { get; private set; }
        public int Seconds { get; private set; }

        public CHugeDateTimeOffset Negate() => new CHugeDateTimeOffset(-this.Years, -this.Months, -this.Days, -this.Hours, -this.Minutes, -this.Seconds);

        internal static void TestDateTimeOffset()
        {
            Test((new CHugeDateTime(2, 1, 1, 0, 0, 0).AddUnexact(new CHugeDateTimeOffset(0, -1, 0, 0, 0, 0))).Year == 1);
            Test((new CHugeDateTime(2, 1, 1, 0, 0, 0).AddUnexact(new CHugeDateTimeOffset(0, -1, 0, 0, 0, 0))).Month == 12);
        }
    }


    internal struct CDtParts
    {
        public CDtParts(bool aSchaltjahr, UInt64 aJahr, int aMonat, int aTag, int aStunde, int aMinute, int aSekunde, DayOfWeek aDayOfWeek )
        {
            this.Year = aJahr;
            this.Month = aMonat;
            this.Day = aTag;
            this.Hour = aStunde;
            this.Minute = aMinute;
            this.Second = aSekunde;
            this.Schaltjahr = aSchaltjahr;
            this.DayOfWeek = aDayOfWeek;
        }
        internal readonly bool Schaltjahr;
        internal readonly UInt64 Year;
        internal readonly int Month;
        internal readonly int Day;
        internal readonly int Hour;
        internal readonly int Minute;
        internal readonly int Second;
        internal readonly DayOfWeek DayOfWeek;
    }
    public abstract class CHugeDateTimeBase
    {
        protected static int GetInt(int[] a, int i, int d = 0) => i < a.Length ? a[i] : d;

        protected const int SecondsPerMinute = 60;
        protected const int SecondsPerHour = SecondsPerMinute * MinutesPerHour;
        protected const int SecondsPerDay = SecondsPerHour * HoursPerDay;
        protected const int MinutesPerHour = 60;
        protected const int MinutesPerDay = MinutesPerHour * HoursPerDay;
        protected const int HoursPerDay = 24;
        protected const int MonthsPerYear = 12;

        internal static void Test(bool ok)
        {
            if (ok)
            {
                // Test passed.
            }
            else if (false && System.Diagnostics.Debugger.IsAttached)
            { 
                System.Diagnostics.Debugger.Break();
            }
            else
            {
                Debug.LogError("HugeDateTime: Test failed.");
            }
        }
        public static void Test()
        {
            CHugeTimespan.TestTimespan();
            CHugeDateTime.TestDateTime();
            CHugeDateTimeOffset.TestDateTimeOffset();
        }

            protected string Pad(Int64 i, int l = 2, bool aExplicitPlus = false)
        => (i < 0 ? "-" : aExplicitPlus ? "+" : "") + Math.Abs(i).ToString().PadLeft(l, '0');

    }

    public sealed class CHugeDateTime
    :
        CHugeDateTimeBase,
        IComparable<CHugeDateTime>
    {
        #region Equals
        public static bool operator ==(CHugeDateTime lhs, CHugeDateTime rhs)
        {
            return lhs.Value == rhs.Value;
        }
        public static bool operator !=(CHugeDateTime lhs, CHugeDateTime rhs)
        {
            return !(lhs == rhs);
        }
        public override bool Equals(object obj)
        {
            if (obj is CHugeDateTime)
                return this == ((CHugeDateTime)obj);
            return false;
        }
        public override int GetHashCode()
        {
            return this.Value.GetHashCode();
        }
        #endregion
        #region CompareTo
        public int CompareTo(CHugeDateTime rhs)
            => this.Value.CompareTo(rhs.Value);
        #endregion

        private static bool GetSchaltjahr(UInt64 aYear) => (aYear % (UInt64) YearsPerSchaltjahrEpoche) == 0;      

        private const int DaysPerJanuary = 31;
        private const int DaysPerFebruaryNormjahr = 28;
        private const int DaysPerFebruarySchaltjahr = DaysPerFebruaryNormjahr + 1;
        private const int Schalttag = DaysPerJanuary + DaysPerFebruarySchaltjahr;
        private static readonly int[] DayCountOfMonthNormal = new int[] { DaysPerJanuary, DaysPerFebruaryNormjahr, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        private static readonly int[] DayCountOfMonthSchaltjahr = new int[] { DaysPerJanuary, DaysPerFebruarySchaltjahr, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        private static readonly int DaysPerSchaltjahr = DayCountOfMonthSchaltjahr.Sum();
        private static readonly int DaysPerNormjahr = DayCountOfMonthNormal.Sum();
        private static readonly int YearsPerSchaltjahrEpoche = 4;
        private static readonly int DaysPerSchaltjahrEpoche = DaysPerSchaltjahr + DaysPerNormjahr * (YearsPerSchaltjahrEpoche - 1);
        private static readonly int SecondsPerSchaltjahrEpoche = DaysPerSchaltjahrEpoche * SecondsPerDay;
        private static readonly int SecondsPerSchaltjahr = DaysPerSchaltjahr * SecondsPerDay;
        private static readonly int SecondsPerNormjahr = DaysPerNormjahr * SecondsPerDay;

        private static int[] GetDaysPerMonth(bool aSchaltjahr) => aSchaltjahr ? DayCountOfMonthSchaltjahr : DayCountOfMonthNormal;
        private static int[] GetDaysPerMonth(UInt64 aYear) => GetDaysPerMonth(GetSchaltjahr(aYear));

        private static int GetDaysPerMonth(UInt64 aYear, UInt64 aMonth)
            => GetDaysPerMonth(aYear)[aMonth];
        private static int GetDaysPerMonthExt(Int64 aYear, int aMonth)
            => GetDaysPerMonth(GetSchaltjahr(aYear < 0 ? (UInt64) Math.Abs(aYear + 1) : (UInt64) aYear))[aMonth];
        private static UInt64 GetAbsoluteDay(UInt64 v)
            => v / SecondsPerDay;
        private static UInt64 GetSchaltjahrCount(UInt64 v) => (v - 1) / 4;
        private static void TestParts()
        {
            //WriteTestResults();

            /*
            Test(new CHugeDateTime(-2).Value == -2);
            Test(new CHugeDateTime(-1).Value == -1);

            var aAd0001_01_01_00_00_00 = (Int64)(0);
            var aAd0001_01_01_00_00_01 = (Int64)(1);
            var aAd0001_01_01_00_00_59 = (Int64)(59);
            var aAd0001_01_01_00_01_00 = (Int64)(60);
            var aAd01_01_01_00_59_59 = (Int64)(59 * SecondsPerMinute + 59);
            var aAd01_01_01_23_59_59 = (Int64)(23 * SecondsPerHour + 59 * SecondsPerMinute + 59);
            var aAd01_01_02_00_00_00 = (Int64)(SecondsPerDay);
            var aAd01_02_01_00_00_00 = (Int64)(SecondsPerDay * 31);
            var aAd01_02_28_23_59_59 = (Int64)(SecondsPerDay * 31 + SecondsPerDay * 28 - 1);
            var aAd01_03_01_00_00_00 = (Int64)(SecondsPerDay * 31 + SecondsPerDay * 28); 
            var aAd04_02_28_23_59_59 = (Int64)(SecondsPerNormjahr * 3 + SecondsPerDay * 31 + SecondsPerDay * 28 - 1);
            var aAd04_02_29_23_59_59 = (Int64)(SecondsPerNormjahr * 3 + SecondsPerDay * 31 + SecondsPerDay * 29 - 1);
            var aAd04_03_01_00_00_00 = (Int64)(SecondsPerNormjahr * 3 + SecondsPerDay * 31 + SecondsPerDay * 29);
            var aBc01_12_31_23_59_59 = (Int64)(-1);
            new CHugeDateTime(-1);
            var aBc2022_10_21_15_28_33 = (Int64)GetValue(new CDtParts(false, -2022, 10, 21, 15, 28, 33, DayOfWeek.Sunday)); // day of week maybe wrong.
            
           // Test(aBc2022_10_21_15_28_33 == new CHugeDateTime(-2022, 10, 21, 15, 28, 33).Value);
           // Test(!new CHugeDateTime(-2022, 10, 21, 15, 28, 33).IstSchaltjahr);
            //Test(!new CHugeDateTime(aBc2022_10_21_15_28_33).IstSchaltjahr);
            //Test(new CHugeDateTime(aBc01_12_31_23_59_59).IstSchaltjahr);


            var aBc01_01_01_00_00_00 = (Int64)(-SecondsPerSchaltjahrExt);
            var aBc01_02_01_00_00_00 = (Int64)(-SecondsPerSchaltjahrExt + SecondsPerDayExt * 31);
            var aBc01_02_28_23_59_59 = (Int64)(-SecondsPerSchaltjahrExt + SecondsPerDayExt * 31 + SecondsPerDayExt * 28 - 1);
            var aBc01_02_29_23_59_59 = (Int64)(-SecondsPerSchaltjahrExt + SecondsPerDayExt * 31 + SecondsPerDayExt * 29 - 1 );
            var aBc01_03_01_00_00_00 = (Int64)(-SecondsPerSchaltjahrExt + SecondsPerDayExt * 31 + SecondsPerDayExt * 29);
            var aTodoDayOfWeek = (System.DayOfWeek) (-1);
            var aTestCases = new CDtPartsTestCase[]
            {
                new CDtPartsTestCase(aAd0001_01_01_00_00_00, new CDtParts(false, 1, 1, 1, 0, 0, 0, aTodoDayOfWeek)),       // 00
                new CDtPartsTestCase(aAd0001_01_01_00_00_01, new CDtParts(false, 1, 1, 1, 0, 0, 1, aTodoDayOfWeek)),       // 01
                new CDtPartsTestCase(aAd0001_01_01_00_00_59, new CDtParts(false, 1, 1, 1, 0, 0, 59, aTodoDayOfWeek)),      // 02
                new CDtPartsTestCase(aAd0001_01_01_00_01_00, new CDtParts(false, 1, 1, 1, 0, 1, 00, aTodoDayOfWeek)),      // 03
                new CDtPartsTestCase(aAd01_01_01_00_59_59, new CDtParts(false, 1, 1, 1, 0, 59, 59, aTodoDayOfWeek)),     // 04
                new CDtPartsTestCase(aAd01_01_01_23_59_59, new CDtParts(false, 1, 1, 1, 23, 59, 59, aTodoDayOfWeek)),    // 05
                new CDtPartsTestCase(aAd01_01_02_00_00_00, new CDtParts(false, 1, 1, 2, 0, 0, 0, aTodoDayOfWeek)),       // 06
                new CDtPartsTestCase(aAd01_02_01_00_00_00, new CDtParts(false, 1, 2, 1, 0, 0, 0, aTodoDayOfWeek)),       // 07
                new CDtPartsTestCase(aAd01_02_28_23_59_59, new CDtParts(false, 1, 2, 28, 23, 59, 59, aTodoDayOfWeek)),   // 08
                new CDtPartsTestCase(aAd01_03_01_00_00_00, new CDtParts(false, 1, 3, 1, 0, 0, 0, aTodoDayOfWeek)),       // 09
                new CDtPartsTestCase(aAd04_02_28_23_59_59, new CDtParts(true, 4,2,28, 23,59,59, aTodoDayOfWeek)),        // 10
                new CDtPartsTestCase(aAd04_02_29_23_59_59, new CDtParts(true, 4,2,29, 23,59,59, aTodoDayOfWeek)),        // 11
                new CDtPartsTestCase(aAd04_03_01_00_00_00, new CDtParts(true, 4,3,1, 0,0,0, aTodoDayOfWeek)),            // 12
                new CDtPartsTestCase(aBc01_12_31_23_59_59, new CDtParts(true, -1,12,31, 23,59,59, aTodoDayOfWeek)),      // 13



                new CDtPartsTestCase(aBc01_01_01_00_00_00, new CDtParts(true, -1,1,1, 0,0,0, aTodoDayOfWeek)),           // 14
                new CDtPartsTestCase(aBc01_02_01_00_00_00, new CDtParts(true, -1,2,1, 0,0,0, aTodoDayOfWeek)),           // 15
                new CDtPartsTestCase(aBc01_02_28_23_59_59, new CDtParts(true, -1,2,28, 23,59,59, aTodoDayOfWeek)),       // 16
                new CDtPartsTestCase(aBc01_02_29_23_59_59, new CDtParts(true, -1,2,29, 23,59,59, aTodoDayOfWeek)),       // 17
                new CDtPartsTestCase(aBc01_03_01_00_00_00, new CDtParts(true, -1,3,1, 0,0,0, aTodoDayOfWeek)),           // 18
                                                                                                                             
                new CDtPartsTestCase(aBc2022_10_21_15_28_33, new CDtParts(true,-2022,10,21,15,28,33, aTodoDayOfWeek)),      // 19                             

            };

            foreach(var aTestCaseIdx in Enumerable.Range(0, aTestCases.Length))
            {
                var aTestCase = aTestCases[aTestCaseIdx];
                var aTestCaseParts = aTestCase.Item2;
                var aTestCaseValue = aTestCase.Item1;

                {
                    var aHaveParts = GetParts(aTestCaseValue);
                    var aSchaltJahrOk = aHaveParts.Schaltjahr == aTestCaseParts.Schaltjahr;
                    var aYearOk = aHaveParts.Year == aTestCaseParts.Year;
                    var aMonthOk = aHaveParts.Month == aTestCaseParts.Month;
                    var aDayOk = aHaveParts.Day == aTestCaseParts.Day;
                    var aHourOk = aHaveParts.Hour == aTestCaseParts.Hour;
                    var aMinuteOk = aHaveParts.Minute == aTestCaseParts.Minute;
                    var aSekundeOk = aHaveParts.Second == aTestCaseParts.Second;
                    Test(aSchaltJahrOk);
                    Test(aYearOk);
                    Test(aMonthOk);
                    Test(aDayOk);
                    Test(aHourOk);
                    Test(aMinuteOk);
                    Test(aSekundeOk);
                }
                { // ReverseTest
                    var aCalcualatedValue = (int)GetValue(aTestCaseParts);
                    var aValueOk = aCalcualatedValue == aTestCaseValue;
                    Test(aValueOk);

                }
            }*/
        }
        private static UInt64 CheckInternalYear(Int64 y)
            => y >= 0 ? (UInt64)y : throw new CYearTooSmallExc();
        private static UInt64 GetInternalYear(Int64 aYear)
            => CheckInternalYear(aYear + Convert.ToInt64(YearOffsetConst) + (aYear < 1 ? 1 : 0));
        private static UInt64 GetValue(Int64 aYear, int aMonth, int aDay, int aHour, int aMinute, int aSecond)
        {
            if (aYear == 0)
                throw new CNoYearZeroExc();
            if (aMonth < 1)
                throw new CMonthTooSmallExc();
            if (aMonth > 12)
                throw new CMonthTooBigExc();
            if (aDay < 1)
                throw new CDayTooSmallExc();
            if (aHour < 0)
                throw new CHourTooSmallExc();
            if (aHour >= HoursPerDay)
                throw new CHourTooBigExc();
            if (aMinute < 0)
                throw new CMinuteTooSmallExc();
            if (aMinute >= MinutesPerHour)
                throw new CMinuteTooBigExc();
            if (aSecond < 0)
                throw new CSecondTooSmallExc();
            if (aSecond >= SecondsPerMinute)
                throw new CSecondTooBigExc();
            var yr = GetInternalYear(aYear);
            var mo = aMonth;
            var to = aDay;
            var v = (UInt64)0;
            var sje = yr / (UInt64)YearsPerSchaltjahrEpoche;
            var sjr = yr % (UInt64)YearsPerSchaltjahrEpoche;
            v += sje * (UInt64)SecondsPerSchaltjahrEpoche;
            if (sjr > 0) { --sjr; v += (UInt64)SecondsPerSchaltjahr; }
            if (sjr > 0) { --sjr; v += (UInt64)SecondsPerNormjahr; }
            if (sjr > 0) { --sjr; v += (UInt64)SecondsPerNormjahr; }
            var sj = GetSchaltjahr(yr);
            var m = (UInt64)aMonth - 1;
            var dpm = GetDaysPerMonth(sj);
            if (aDay > dpm[m])
                throw new CDayTooBigExc();
            for (var mi = (UInt64)0; mi < m; ++mi)
            {
                v += (UInt64)dpm[mi] * SecondsPerDay;
            }
            var d = (UInt64)aDay - 1;
            v += d *  (UInt64)SecondsPerDay;
            var h = (UInt64) aHour;
            v += h * (UInt64)SecondsPerHour;
            var n = (UInt64)aMinute;
            v += n * (UInt64)SecondsPerMinute;
            var s = (UInt64)aSecond;
            v += s;
            var r = v; 
            return r;
        }

        public abstract class CExc : Exception { }
        public sealed class CInternalErrorExc : CExc { }
        public sealed class CNoYearZeroExc : CExc { }
        public sealed class CYearTooSmallExc : CExc { }
        public sealed class CMonthTooBigExc : CExc { }
        public sealed class CMonthTooSmallExc : CExc { }
        public sealed class CDayTooSmallExc : CExc { }
        public sealed class CDayTooBigExc : CExc { }

        public sealed class CHourTooSmallExc : CExc { }
        public sealed class CHourTooBigExc : CExc { }
        public sealed class CMinuteTooSmallExc : CExc { }
        public sealed class CMinuteTooBigExc : CExc { }
        public sealed class CSecondTooSmallExc : CExc { }
        public sealed class CSecondTooBigExc : CExc { }
        private static DayOfWeek GetDayOfWeek(UInt64 aValue)
        {
            var o1 = SecondsOffset / SecondsPerDay;
            var o2 = o1 / 7 * 7;
            var o = o2;

            var v0 = aValue;
            var v1 = v0 / SecondsPerDay;
            var v2 = v1 % 7;
            var v = v2;

            var s1 = o + v + DayOffset;
            var s2 = s1 % 7;
            var s = s2;

            var d = (DayOfWeek)s;
            return d;
        }

        private static CDtParts GetParts(UInt64 aValue)
        {
            var aDayOfWeek = GetDayOfWeek(aValue);         
            var v = aValue;
            var sje = v / (UInt64)SecondsPerSchaltjahrEpoche;
            var y = sje * (UInt64)YearsPerSchaltjahrEpoche;
            v -= sje * (UInt64)SecondsPerSchaltjahrEpoche;
            for (var yi = (UInt64)0; yi < (UInt64)YearsPerSchaltjahrEpoche; ++yi)
            {
                var sjj = (y);
                var sj = GetSchaltjahr(sjj);
                var dpm = GetDaysPerMonth(sj);
                for (var m = (UInt64)0; m < MonthsPerYear; ++m)
                {
                    var d2 = (UInt64)dpm[m];
                    var s1 = d2 * SecondsPerDay;
                    if (s1 > v)
                    {
                        var d = v / SecondsPerDay;
                        v -= d * SecondsPerDay;
                        var h = v / SecondsPerHour;
                        v -= h * SecondsPerHour;
                        var min = v / SecondsPerMinute;
                        v -= min * SecondsPerMinute;
                        var s = v;
                        var yr = y;
                        var ps = new CDtParts(sj, yr, (int)m + 1, (int)d + 1, (int)h, (int)min, (int)s, aDayOfWeek);
                        return ps;
                    }
                    else
                    {
                        v -= s1;
                    }
                }
                ++y;
            }
            throw new CInternalErrorExc();
        }

        public UInt64 Value { get; private set; }
        private readonly CDtParts DtParts;

        private CHugeDateTime(UInt64 aValue)
        {
            this.Value = aValue;
            this.DtParts = GetParts(aValue);
        }
        public CHugeDateTime(Int64 aYear, int aMonth, int aDay, int aHour, int aMinute, int aSecond)
        : this(GetValue(aYear, aMonth, aDay, aHour, aMinute, aSecond))
        {

        }

        public CHugeDateTime(int aYear, int aMonth, int aDay, int aHour, int aMinute, int aSecond)
            : this((Int64)aYear, aMonth, aDay, aHour,aMinute, aSecond)
        {

        }

        public bool IstSchaltjahr => this.DtParts.Schaltjahr;
        public CHugeDateTime() : this(1,1,1,0,0,0) { }

        public static void TestDateTime()
        {
            Test(new CHugeDateTime().Year == 1);
            Test(new CHugeDateTime().Month == 1);
            Test(new CHugeDateTime().Day == 1);
            Test(new CHugeDateTime().Hour == 0);
            Test(new CHugeDateTime().Minute == 0);
            Test(new CHugeDateTime().Second == 0);
            Test(new CHugeDateTime().IsAd);
            Test(new CHugeDateTime().DayOfWeek == DayOfWeek.Monday);
            Test(new CHugeDateTime().AddSeconds(SecondsPerNormjahr * 3).IstSchaltjahr);
            Test(new CHugeDateTime().AddSeconds(-1).IstSchaltjahr);
            Test(new CHugeDateTime(1, 1, 1, 0, 0, 0).IsAd);
            Test(new CHugeDateTime(1, 1, 1, 0, 0, 0).Year == 1);
            Test(new CHugeDateTime(2, 1, 1, 0, 0, 0).Year == 2);
            Test(new CHugeDateTime(3, 1, 1, 0, 0, 0).Year == 3);
            Test(new CHugeDateTime(4, 1, 1, 0, 0, 0).Year == 4);
            Test(new CHugeDateTime(-1, 1, 1, 0, 0, 0).Year == -1);
            Test(new CHugeDateTime(-1, 1, 1, 0, 0, 0).IsBc);
            Test(new CHugeDateTime(-1, 12, 31, 0, 0, 0).DayOfWeek ==  DayOfWeek.Sunday);      
            Test(new CHugeDateTime(-5, 1, 1, 0, 0, 0).Year == -5);
            Test(new CHugeDateTime(2022, 10, 21, 15, 28, 33).Year == 2022);
            Test(new CHugeDateTime(2022, 10, 21, 15, 28, 33).Month == 10);
            Test(new CHugeDateTime(2022, 10, 21, 15, 28, 33).Day == 21);
            Test(new CHugeDateTime(2022, 10, 21, 15, 28, 33).Hour == 15);
            Test(new CHugeDateTime(2022, 10, 21, 15, 28, 33).Minute == 28);
            Test(new CHugeDateTime(2022, 10, 21, 15, 28, 33).Second == 33);
            Test(new CHugeDateTime(-2022, 1, 1, 0, 0, 0).Year == -2022);
            Test(new CHugeDateTime(-2022, 10, 21, 15, 28, 33).Year == -2022);
            Test(new CHugeDateTime(-2022, 10, 21, 15, 28, 33).Month == 10);
            Test(new CHugeDateTime(-2022, 10, 21, 15, 28, 33).Day == 21);
            Test(new CHugeDateTime(-2022, 10, 21, 15, 28, 33).Hour == 15);
            Test(new CHugeDateTime(-2022, 10, 21, 15, 28, 33).Minute == 28);
            Test(new CHugeDateTime(-2022, 10, 21, 15, 28, 33).Second == 33);
        }

        public Int64 YearAbs => Math.Abs(this.Year);
        private Int64 FixYear(Int64 y) => y - (y <= 0 ? 1 : 0);
        public Int64 Year => FixYear(Convert.ToInt64(this.DtParts.Year) - Convert.ToInt64(YearOffset));
        public bool IsBc => this.Year < 0;
        public bool IsAd => this.Year > 0;
        public override string ToString()
            => this.GregorianPrefix + " "
            + Pad(this.Day) + "."
            + Pad(this.Month) + "."
            + Pad(this.YearAbs, 4) + " "
            + Pad(this.Hour) + ":"
            + Pad(this.Minute) + ":"
            + Pad(this.Second)
            ;
        public string GregorianPrefix => this.IsBc ? "BC" : this.IsAd ? "AD" : throw new CInternalErrorExc();
        public int Month => this.DtParts.Month;
        public int Day => this.DtParts.Day;
        public int Hour => this.DtParts.Hour;
        public int Minute => this.DtParts.Minute;
        public int Second => this.DtParts.Second;
        public DayOfWeek DayOfWeek => this.DtParts.DayOfWeek;

        public static CHugeDateTime operator +(CHugeTimespan ts, CHugeDateTime dt)
        {
            return dt + ts;
        }
        public static CHugeDateTime operator +(CHugeDateTime dt, CHugeTimespan ts)
        {
            return dt.Add(ts);
        }
        public CHugeDateTime Add(CHugeTimespan ts)
            => new CHugeDateTime( this.Value + (UInt64)ts.Value);
        public CHugeDateTime AddSeconds(Int64 aSeconds) => this.Add(CHugeTimespan.FromSeconds(aSeconds));
        public CHugeDateTime AddMinutes(Int64 aMinutes) => this.Add(CHugeTimespan.FromMinutes(aMinutes));
        public CHugeDateTime AddHours(Int64 aHours) => this.Add(CHugeTimespan.FromHours(aHours));
        public CHugeDateTime AddDays(Int64 aDays) => this.Add(CHugeTimespan.FromDays(aDays));
        public CHugeDateTime AddMonths(int aMonths)
            => this.AddMonthsUnexact(aMonths);
        private Int64 FixGregorianYear0(Int64 y)
            => y + (y <= 0 ? -1 : 0);

        /// <summary>
        /// Adds months to current date where we really increment the month by reserving the day except a day becomes invalid.
        /// then the routine decrements the day.
        /// </summary>
        /// <param name="aMonths"></param>
        /// <returns></returns>
        public CHugeDateTime AddMonthsExact(int aMonths)
            => this.Add(this.GetMonthsTimespan(aMonths));
        public CHugeDateTime AddMonthsUnexact(int aMonths)
        {
            var mc = (Int64)aMonths;
            var mo = (Int64)this.Month - 1;
            var mn1 = mo + mc;
            var mya1 = mn1 / (Int64)MonthsPerYear;
            var mn2 = mn1 % (Int64)MonthsPerYear;
            var mya2 = mn2 < 0 ? -1 : 0;
            var mya = mya1 + mya2;
            var mn = (Int64) (mn2 < 0 ? (Int64)MonthsPerYear + mn2 : mn2);
            var aNewMonth = (int)(mn + 1);
            var yo = (Int64)this.Year;
            var yn1 = yo + mya;
            var yn = yn1;
            var yn2 = FixGregorianYear0(yn1); 
            var aNewYear = (int) yn2;
            var aNewDay = Math.Min(this.Day, (int)GetDaysPerMonthExt(yn, (int) mn)); // Tage reduzieren, falls es den tag nicht gibt. (Etwas fuzzy, da beim addieren das momentan nicht berücksichtigt wird- siehe GetMonthsTimespan not implemented.)
            var dt = new CHugeDateTime(aNewYear, aNewMonth, aNewDay, this.Hour, this.Minute, this.Second);
            return dt;
        }
        public CHugeDateTime AddYearsUnexact(int aYears)
            => this.AddYearsUnexact(Convert.ToInt64(aYears));

        /// <summary>
        /// Adds years to current date by modifying only year and keeping the rest where possible.
        /// in case of leap years, the 29th of february will be decremented.
        /// </summary>
        /// <param name="aYears"></param>
        /// <returns></returns>
        public CHugeDateTime AddYearsUnexact(Int64 aYears)
        {
            var y = FixGregorianYear0(this.Year + aYears);
            var m = this.Month;
            var d1 = this.Day;
            var dpm = GetDaysPerMonth(GetInternalYear(y), Convert.ToUInt64(m-1));
            var d2 = Math.Min(d1, dpm);
            var d = d2;
            var h = this.Hour;
            var n = this.Minute;
            var s = this.Second;
            var dt = new CHugeDateTime(y, m, d, h, n, s);
            return dt;
        }
        public CHugeDateTime AddYears(int aYears) 
            => this.Add(this.GetYearsTimespan(aYears));
        public CHugeTimespan GetYearsTimespan(int aYears) 
            => this.GetMonthsTimespan((int)(aYears * MonthsPerYear));
        public CHugeTimespan GetMonthsTimespan(int aMonths)
            => throw new NotImplementedException();
        public CHugeDateTime Add(CHugeDateTimeOffset rhs)
            => this.AddSeconds(rhs.Seconds).AddMinutes(rhs.Minutes).AddHours(rhs.Hours).AddDays(rhs.Days).AddMonths(rhs.Months).AddYears(rhs.Years);
        public CHugeDateTime AddUnexact(CHugeDateTimeOffset rhs)
            => this.AddSeconds(rhs.Seconds).AddMinutes(rhs.Minutes).AddHours(rhs.Hours).AddDays(rhs.Days).AddMonthsUnexact(rhs.Months).AddYearsUnexact(rhs.Years);
        public CHugeDateTime SubtractUnexact(CHugeDateTimeOffset rhs)
        {
            return this.AddUnexact(rhs.Negate());
        }
        public static CHugeDateTime operator +(CHugeDateTimeOffset offs, CHugeDateTime dt)
        {
            return dt.Add(offs);
        }
        public static CHugeDateTime operator +(CHugeDateTime dt, CHugeDateTimeOffset offs)
        {
            return dt.Add(offs);
        }

        public static CHugeDateTime operator -(CHugeDateTime lhs, CHugeTimespan rhs)
        {
            return lhs.Subtract(rhs);
        }
        public CHugeDateTime Subtract(CHugeTimespan rhs)
            => this.Add(rhs.Negate());

        #region Bc-Handling 
        private static readonly UInt64 FirstAdValueSchaltjahrEppocheOffsetMax = (UInt64) (Int64.MaxValue / SecondsPerSchaltjahrEpoche);
        private static readonly UInt64 FirstAdValueSchaltjahrEppocheOffset = FirstAdValueSchaltjahrEppocheOffsetMax;
        private static readonly UInt64 SecondsOffset = (UInt64)SecondsPerSchaltjahrEpoche * FirstAdValueSchaltjahrEppocheOffset;
        private const int DayOffset = 6;
        private static UInt64 YearOffset = FirstAdValueSchaltjahrEppocheOffset * (UInt64)YearsPerSchaltjahrEpoche;
        private static readonly UInt64 YearOffsetConst = FirstAdValueSchaltjahrEppocheOffset * (UInt64) YearsPerSchaltjahrEpoche;
        public static readonly CHugeDateTime MinValue = new CHugeDateTime(UInt64.MinValue);
        public static readonly CHugeDateTime MaxValue = new CHugeDateTime(UInt64.MaxValue);
        #endregion
    }



    public sealed class CHugeTimespan : CHugeDateTimeBase
    {
        public static CHugeTimespan MinValue = new CHugeTimespan(Int64.MinValue);
        public static CHugeTimespan MaxValue = new CHugeTimespan(Int64.MaxValue);
        internal Int64 Value { get; private set; }
        private UInt64 ValueInternal => (UInt64)this.Value;
        public CHugeTimespan()
        {
        }
        public CHugeTimespan(Int64 aValue)
        {
            this.Value = aValue;
        }
        public static CHugeTimespan operator +(CHugeTimespan lhs, CHugeTimespan rhs)
        {
            return new CHugeTimespan(lhs.Value + rhs.Value);
        }
        public static CHugeTimespan FromSeconds(Int64 aSeconds) => new CHugeTimespan(aSeconds);
        public static CHugeTimespan FromMinutes(Int64 aMinutes) => new CHugeTimespan(aMinutes * (Int64)SecondsPerMinute);
        public static CHugeTimespan FromHours(Int64 aHours) => new CHugeTimespan(aHours * (Int64)SecondsPerHour);
        public static CHugeTimespan FromDays(Int64 aDays) => new CHugeTimespan(aDays * (Int64)SecondsPerDay);

        public int Seconds => (int)(this.ValueInternal % SecondsPerMinute);
        public int Minutes => (int)((this.ValueInternal / SecondsPerMinute) % SecondsPerMinute);
        public int Hours => (int)((this.ValueInternal / SecondsPerHour) % HoursPerDay);
        public Int64 Days => (this.Value / (Int64)SecondsPerDay);
        public CHugeTimespan AddSeconds(Int64 s) => new CHugeTimespan(this.Value + s);
        internal static void TestTimespan()
        {
            var aGetSeconds = new Func<int, int, int, int, int>(delegate (int s, int m, int h, int d)
            {
                return s + m * 60 + h * 60 * 60 + d * 60 * 60 * 24;
            });

            for (var aDayOffset = 0; aDayOffset < 2; ++aDayOffset)
            {
                var aMin = 0;
                var aMax = Math.Pow(2, 4);
                for (var i = aMin; i < aMax; ++i)
                {
                    var s = (i & 1) >> 0;
                    var m = (i & 2) >> 1;
                    var h = (i & 4) >> 2;
                    var d = (i & 8) >> 3;
                    var aSeconds = aGetSeconds(s, m, h, d + aDayOffset);
                    var aTimeSpan = CHugeTimespan.FromSeconds(aSeconds);
                    //if (i == 1)
                    //    System.Diagnostics.Debugger.Break();
                    //Test(aTimeSpan.Seconds == s);
                    //Test(aTimeSpan.Minutes == m);
                    //Test(aTimeSpan.Hours == h);
                    //Test(aTimeSpan.Days == d + aDayOffset);
                }
            }
        }
        public CHugeTimespan Negate() => new CHugeTimespan(-this.Value);
    }

}
