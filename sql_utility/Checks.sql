-- controllo che ogni stagione abbia 38 partite 
select stagione,count(*)
from PARTITE_SERIE_A PSA 
group by stagione;

-- creo le giornate sfruttando il fatto che in ogni giornata giocano 10 partite e controllo che ogni giornata ne abbia 10 per ogni 
-- squadra, ovvero che non ci siano partite rimandate che sballano il risultato
with giornate As(
SELECT
	CEIL( (ROW_NUMBER() OVER (PARTITION BY stagione ORDER BY Date))/10) AS Giornata,
    Date,
    Stagione,
    HomeTeam,
    AwayTeam
FROM
    PARTITE_SERIE_A)
Select 
count(HomeTeam) casa_per_giornata,
count(AwayTeam) trasferta_per_giornata,
Stagione,
giornata
from giornate
group by stagione,giornata
having casa_per_giornata <> 10 and trasferta_per_giornata <> 10;

-- controllo che ogni stagione abbia 10 giornate
with giornate As(
SELECT
	CEIL( (ROW_NUMBER() OVER (PARTITION BY stagione ORDER BY Date))/10) AS Giornata,
    Date,
    Stagione,
    HomeTeam,
    AwayTeam
FROM
    PARTITE_SERIE_A)
select 
max(giornata),
stagione 
from giornate
group by stagione;