import datetime

from csep import query_gcmt

from floatcsep.utils.accessors import query_sed



start_time = datetime.datetime(2025,4,4,0,0,0)
end_time = datetime.datetime(2025,5,11,0,0,0)

cat = query_sed(start_time, end_time, min_magnitude=2.5)


cat.plot(show=True)