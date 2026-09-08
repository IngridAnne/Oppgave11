[] Finne en API med masse aksjeprisdata
[] Finne mulig korrelerte aksjepar ved å bruke API-en
[] Beregne forholdet mellom prisene
[] Beregne deretter rolling mean og standardavvik
    - Når spreaden beveger seg mer enn eksempelvis to standardavvik fra normalen, åpnes en posisjon
    - Når spreaden normaliseres, lukkes den
[] Legge inn transaksjonskostander for å sjekke om strategien lønner seg

# Status
Laget et utkast for å automatisk hente data og regne ut korrelasjon, samt rangere kandidat-parene
## Next step
Bruke yfinance api til å hente real-time data og kjøre tester for statistisk arbitrasje (kjøp den relativt billige aksjen, selg den relativt dyre)
