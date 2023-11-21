-- controllo che ogni stagione abbia 38 partite 
select stagione,count(*)
from PARTITE_SERIE_A PSA 
group by stagione;