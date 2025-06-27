# bike-UAE
Center for Interacting Urban Networks @ NYU Abu Dhabi Data Analysis Internship

This is where you'll find all the technical analysis done for the biking infrusturcture research project I've done this summer for my Internship with NYU @ Abu Dhabi. My director supervisor is Xander Cristou, Transportation Expert, and the data being analyzed is a survey completed by UAE residents and Strava Metro data.

##########################################################################

In '2025-06-16_Bike-UAE_Survey-Data-Cleaning.py' you can find the data cleaning process I took for organizing the survey data (LABELS-UAE Bicycle & Scooter Survey - CITIES_June 16, 2025_16.12.csv). The cleaned excel file is 'Bike-UAE_Basic-Cleaned.xlsc'.

#######
Cleaned Bike-UAE Data Feature Descriptions // 2025-06-26 // Max Moran
#######

Starting User Info
Finished - Boolean - Whether the user finished the survey or not
UserLang - Factor :  EN, AR - Which language the user chose to take the survey with
Progress - Float - How far through the survey the user completed
Duration - Float - How long it took the user to enter their answers

Transport Info
TranspoType - Factor :  Bicycle, Electric Bicycle, Electric Scooter - Which form of micro-transport the user uses the most
Type-BMXBike - Boolean -  If the user owns a BMX Bike
Type-CyclocrossBike - Boolean - If the user owns a Cyclocross Bike
Type-ElectricBike - Boolean -  If the user owns a Electric Bike
Type-FoldingBike - Boolean -  If the user owns a Folding Bike
Type-HybridBike - Boolean -  If the user owns a Hybrid Bike
Type-MountainBike - Boolean -  If the user owns a Mountain Bike
Type-RoadBike - Boolean -  If the user owns a Road Bike
Not sure - Boolean -  If the user is not sure what kind of bike they own
BikeType-Other - Boolean -  If the user owns another kind of Bike
BikeType-Other-Text - String - User text entry for what other type of bike they own

TranspoType-HowOften - Factor : Daily, 4-6 times a week, 2-3 times a week, Once a week, A few times a month, Once a month, A few times a year - How often the user rides
RiderType - Factor :  I ride professionally, I am a fast and fearless rider, I am a confident but casual rider, I am interested in riding, but I have concerns, Other - What type of rider the user is
RiderType-Other-Text - String - User text entry for what type of rider the user is
Bikeshare - Factor : Yes, always, Yes, many times, Yes, sometimes, No - Does the user use bikeshare services in the UAE

Why-TourdeFrance - Boolean - If the user rides because they want to be the next Tour de France winner
Why-Commuting - Boolean - If the user rides because they commute or use it as a mode of transport
Why-Racing - Boolean - If the user rides because they do competitive racing
Why-Fitness - Boolean - If the user rides because they want to stay fit
Why-Leisure - Boolean - If the user rides because they enjoy the recreational and leisure aspect
TranspoType-Why-Other - Boolean - If the user rides for another reason
TranspoType-Why-Other-Text - String - User text entry for why they ride

AltMicro-Bike - Boolean - If the user also rides a bike
AltMicro-EBike - Boolean - If the user also rides an ebike
AltMicro-EScooter - Boolean - If the user also rides an electric scooter
AltMicro-ESkateboard - Boolean - If the user also rides an electric skateboard
AltMicro-Camel - Boolean - If the user also rides a camel
AltMicro-No - Boolean - If the user does not ride any other forms of micro mobility

Rider Info
Gender - Factor : Male, Female, Prefer not to say - What gender the user is
Age - Factor :  Under 18, 18-24 years old, 25-34 years old, 35-44 years old, 45-54 years old, 55-64 years old, 65+ years old - What age range is the user in
Country - Factor : Afghanistan, Algeria, Argentina, Armenia, Australia, Austria, Bahrain, Bangladesh, Belgium, Brazil, Brunei Darussalam, Bulgaria, Canada, Colombia, Congo, Republic of the..., Croatia, Czech Republic, Denmark, Ecuador, Egypt, Ethiopia, Finland, France, Georgia, Germany, Ghana, Greece, Hungary, India, Indonesia, Iran, Iraq, Ireland, Italy, Japan, Jordan, Kazakhstan, Kenya, Kosovo, Kyrgyzstan, Latvia, Lebanon, Malaysia, Mauritius, Mongolia, Morocco, Nepal, Netherlands, New Zealand, Norway, Oman, Pakistan, Palestine, Paraguay, Peru, Philippines, Poland, Portugal, Romania, Russian Federation, Serbia, Singapore, Slovakia, Slovenia, South Africa, South Korea, Spain, Sri Lanka, Sudan, Sweden, Switzerland, Syria, Tanzania, Tunisia, Turkey, Uganda, Ukraine, United Arab Emirates, United Kingdom of Great Britain and Northern Ireland, United States of America, Uzbekistan - Which country the user is from
Education - Factor : Some Primary, Primary School, Some Secondary, Secondary School, Vocational or Similar, Some University but no degree, University Associate Degree, University Bachelors Degree, Graduate or professional degree (MA, MS, MBA, PhD, JD, MD, DDS), Prefer not to say, Other - What level of education the user has completed
Education-Other-Text - String - User text entry for what type of education they have

OwnBike - Factor :  Yes, No - Does the user own a bike
Helmet - Factor :  ‘Yes, always’, ‘Yes, many times’, ‘Yes, sometimes’, ‘No’ - Does the rider use a helmet when riding
Earbuds - Factor :  ‘Yes, always’, ‘Yes, many times’, ‘Yes, sometimes’, ‘No’ - Does the rider use earbuds when riding
Phones - Factor :  ‘Yes, always’, ‘Yes, many times’, ‘Yes, sometimes’, ‘No’ - Does the rider use a phone when riding
TrackRides - Factor :  ‘Yes, always’, ‘Yes, many times’, ‘Yes, sometimes’, ‘No’ - Does the rider use a tracking platform when riding

EmploymentStatus - Factor :  Working full-time, Working part-time, Unemployed and looking for work, A homemaker or stay-at-home parent, Student, Retired, Other - What best describes the user’s employment status over the last three months
Salary - Factor :  Less than 5k, 5-10k, 10-15k, 15-20k, 20-25k, 25-30k, More than 30k, Prefer not to share - What is the user’s monthly salary range
DurationInUAE - Float : 1 - 40+ - How many years has the user called the UAE home
YearsRiding - Factor :  I am a new rider, 1-5 years, 5-10 years, 10-15 years, 15-20 years, More than 20 years, Since childhood - How many years has the user been riding

January - Boolean - If the user rides in January
February - Boolean - If the user rides in February
March - Boolean - If the user rides in March
April - Boolean - If the user rides in April
May - Boolean - If the user rides in May
June - Boolean - If the user rides in June
July - Boolean - If the user rides in July
August - Boolean - If the user rides in August
September - Boolean - If the user rides in September
October - Boolean - If the user rides in October
November - Boolean - If the user rides in November
December - Boolean - If the user rides in December

KidsCycle - Factor :  Yes, Maybe, No (please explain why not) - If the user had kids, do they let them ride to school
KidsCycle-No-Text - String - User text entry for why they don’t let their kids ride
RegularCommute - Factor :  Yes, No - Does the user have a regular commute

NoRideReason-Cultural - Boolean - If cultural reasons prevent the user from riding to work
NoRideReason-Distance - Boolean - If distance reasons prevent the user from riding to work
NoRideReason-Judgment - Boolean - If fear of judgment reasons prevent the user from riding to work
NoRideReason-NoPaths - Boolean - If lack of cycle paths reasons prevent the user from riding to work
NoRideReason-NoFacilities - Boolean - If lack of facilities at work reasons prevent the user from riding to work
NoRideReason-TooLong - Boolean - If too much time commuting reasons prevent the user from riding to work
NoRideCommute-Other - Boolean - If another reason prevent the user from riding to work
NoRideCommute-Other-Text - String - User text entry for why the user does not ride to work

RideWith-Myself - Boolean - If the user rides by themselves
RideWith-CyclingClub - Boolean - If the user rides with a cycling club
RideWith-Family - Boolean - If the user rides with family
RideWith-Friends - Boolean - If the user rides with friends

LSI-ParticpationInEvents - Boolean - If the user purchasing a bicycle has allowed them to participate in cycling events
LSI-ReduceCarbonFootprint - Boolean - If the user purchasing a bicycle has helped them reduce their carbon footprint
LSI-ImprovedPhysicalFitness - Boolean - If the user purchasing a bicycle has improved their physical fitness and overall health
LSI-EasierDailyCommute - Boolean - If the user purchasing a bicycle has made their daily commute easier and more efficient
LSI-LifestyleNoneOfAbove - Boolean - If the user purchasing a bicycle has not affected them in any of the other ways listed
LSI-NewWayToExplore - Boolean - If the user purchasing a bicycle has provided them with a new way to explore and enjoy the UAE
LifestyleInfluence-Other - Boolean - If the user purchasing a bicycle has affected them an another way that isn’t listed
LifestyleInfluence-Other-Text - String - User text entry for what ways riding has influenced their lifestyle

Riding Area Info
Comfort-CityStreets - Boolean - If the user feels comfortable riding on city streets
Comfort-CyclePaths - Boolean - If the user feels comfortable riding on cycle paths
Comfort-CycleTracks - Boolean - If the user feels comfortable riding on cycle tracks
Comfort-Highways - Boolean - If the user feels comfortable riding on highways
Comfort-NeighborhoodRoads - Boolean - If the user feels comfortable riding on neighborhood roads
Comfort-Parks - Boolean - If the user feels comfortable riding in parks
Comfort-Sidewalks - Boolean - If the user feels comfortable riding on sidewalks

Chal-ChargingAccess - Boolean - If the rider faces charging access as a challenge
Chal-CulturalNorms - Boolean - If the rider faces cultural norms as a challenge
Chal-ExtremeWeather - Boolean - If the rider faces extreme weather as a challenge
Chal-EquipmentCost - Boolean - If the rider faces equipment costs as a challenge
Chal-Judgement - Boolean - If the rider faces judgement as a challenge
Chal-NoCycleLanes - Boolean - If the rider faces no cycling lanes as a challenge
Chal-NoCycleParking - Boolean - If the rider faces no cycle parking as a challenge
Chal-PersonalSafetyConcerns - Boolean - If the rider faces personal safety concerns as a challenge
Chal-PetrolPrices - Boolean - If the rider faces rising petrol prices as a challenge
Chal-RoadSafetyConcerns - Boolean - If the rider faces road safety concerns as a challenge
Challenges-Other - Boolean - If the rider faces other variables as a challenge
Challenges-Other-Text - String - User text entry for what challenge(s) they experience as a rider

ExpImp-IndoorCyclingAccess - Boolean - If the user’s riding experience can be improved by indoor cycling access
ExpImp-Awareness - Boolean - If the user’s riding experience can be improved by awareness for drivers and cyclists
ExpImp-BetterRoads - Boolean - If the user’s riding experience can be improved by better roads from from obstacles
ExpImp-BetterSignage - Boolean - If the user’s riding experience can be improved by better signage and road markings
ExpImp-MoreBikeRentals - Boolean - If the user’s riding experience can be improved by more bike rental services/stations
ExpImp-MoreCyclingLanes - Boolean - If the user’s riding experience can be improved by more cycling lanes
ExpImp-MoreDedicatedTracks - Boolean - If the user’s riding experience can be improved by dedicated cycling tracks
ExpImp-MoreShadedAreas - Boolean - If the user’s riding experience can be improved by more shaded areas
ExperienceImprovement-Other - Boolean - If the user’s riding experience can be improved by other things
ExperienceImprovement-Other-Text - String - User text entry for what can be improved about the riding experience

Shop Info
CSI-SpendTimeWithFamily - Boolean - If the user has had cycling shops contribute to their social life by encouraging them to spend more time with family
CSI-ConnectedGrowingCulture - Boolean -  If the user has had cycling shops contribute to their social life by connecting them to cycling culture
CSI-CharityCyclingEvents - Boolean -  If the user has had cycling shops contribute to their social life by inspiring them to take part in cycling events
CSI-CyclingGroupsAndFriends - Boolean -  If the user has had cycling shops contribute to their social life by introducing them to cycling groups/friends
CSI-WorkshopEventAccess - Boolean -  If the user has had cycling shops contribute to their social life by providing access to workshops, events, or training sessions
CycleShopsInfluence-Other - Boolean -  If the user has had cycling shops contribute to their social life by another reason
CycleShopsInfluence-Other-Text - String - User text entry for how cycling shops have influenced their life

CSB-ReliableMaintenance - Boolean - If the user has had cycling shops benefit them by providing a place for reliable maintenance
CSB-WelcomingCommunity - Boolean - If the user has had cycling shops benefit them by providing a welcoming community
CSB-PremiumGearAccess - Boolean - If the user has had cycling shops benefit them by providing access to premium gear
CSB-HighQualityExpertService - Boolean - If the user has had cycling shops benefit them by providing high quality and expert service
CSB-HealthyMotivation - Boolean - If the user has had cycling shops benefit them by providing motivation to lead a healthier lifestyle
CycleShopsBenefit-Other - Boolean - If the user has had cycling shops benefit them by providing other benefits
CycleShopsBenefit-Other-Text - String - User text entry for what benefits cycling shops have in the UAE

Location Info
Emirate - Factor :  Abu Dhabi, Al Ain, Al Dhafra, Dubai, Sharjah, Ajman, Umm Al Quwain, Fuhairah, Ras al Khaimah - Which Emirate does the user call home
Emirate-Safety - Factor :  1 - very safe, no concerns, 2, 3 - moderately safe, some concerns, 4, 5 - not safe at all, many concerns - How safe the user feels riding in their emirate
Emirate-WhyLikeRiding - String - User text entry for why the user likes riding in their emirate
Emirate-Initiatives - String - User text entry for what initiatives would encourage more people to take up riding in their emirate

AD-AlBateen - Boolean - If the user typically rides in Al Bateen, Abu Dhabi
AD-AlHudayriyatIsland - Boolean - If the user typically rides in Al Hudayriat Island, Abu Dhabi
AD-AlMaryahIsland - Boolean - If the user typically rides in Al Maryah Island, Abu Dhabi
AD-AlRaha - Boolean - If the user typically rides in Al Raha, Abu Dhabi
AD-AlReemIsland - Boolean - If the user typically rides in Al Reem Island, Abu Dhabi
AD-AlSaadiyatIsland - Boolean - If the user typically rides in Al Saadiyat Island, Abu Dhabi
AD-AlShahama - Boolean - If the user typically rides in Al Shahama, Abu Dhabi
AD-AlWathbaCycleTrack - Boolean - If the user typically rides in Al Wathba Cycle Track, Abu Dhabi
AD-TouristClub - Boolean - If the user typically rides in Al Zahiyah/Tourist Club, Abu Dhabi
AD-Corniche - Boolean - If the user typically rides in Corniche, Abu Dhabi
AD-KhalifaCity - Boolean - If the user typically rides in Khalifa City, Abu Dhabi
AD-Musaffah - Boolean - If the user typically rides in Musaffah, Abu Dhabi
AD-YasIsland - Boolean - If the user typically rides in Yas Island, Abu Dhabi

AA-AlAinCycleTrack - Boolean - If the user typically rides in Al Ain Cycle Track, Al Ain
AA-AlJimi - Boolean - If the user typically rides in Al Jimi, Al Ain
Al Mutarid - Boolean - If the user typically rides in Al Mutarid, Al Ain
AA-AlMutawah - Boolean - If the user typically rides in Al Mutawah, Al Ain
AA-CentralDistrict - Boolean - If the user typically rides in Central District, Al Ain
AA-Hili - Boolean - If the user typically rides in Hili, Al Ain
AA-JebelHafeet - Boolean - If the user typically rides in Jebel Hafeet, Al Ain

D-AlQudraCycleTrack - Boolean - If the user typically rides in Al Qudra Cycle Track, Dubai
D-BurDubai - Boolean - If the user typically rides in Bur Dubai, Dubai
D-BusinessBay - Boolean - If the user typically rides in Business Bay, Dubai
D-Deira - Boolean - If the user typically rides in Deira, Dubai
D-DowntownDubai - Boolean - If the user typically rides in Downtown Dubai, Dubai
D-DubaiMarina - Boolean - If the user typically rides in Dubai Marina, Dubai
D-JLT - Boolean - If the user typically rides in JLT (Jumeirah Lake Towers), Dubai
D-JumeirahBeach - Boolean - If the user typically rides in Jumeirah Beach, Dubai
D-MeydanDXBike - Boolean - If the user typically rides in Meydan DXBike, Dubai
D-MushrifPark - Boolean - If the user typically rides in Mushrif Park, Dubai

S-AlBatayehBicycleTrack - Boolean - If the user typically rides in Al Batayeh Bicycle Track, Sharjah
S-AlQasimia - Boolean - If the user typically rides in Al Qasimia, Sharjah
S-AlRiqa - Boolean - If the user typically rides in Al Riqa, Sharjah
S-IndustrialArea - Boolean - If the user typically rides in Industrial Area, Sharjah
S-KhorFakkan - Boolean - If the user typically rides in Khor Fakkan, Sharjah
S-MasaarCyclingTrack - Boolean - If the user typically rides in Masaar Cycling Track, Sharjah
S-SharjahCorniche - Boolean - If the user typically rides in Sharjah Corniche, Sharjah
S-UniversityCity - Boolean - If the user typically rides in University City, Sharjah

RAK-AlJazeeraAlHamra - Boolean - If the user typically rides in Al Jazeera Al Hamra, Ras al Khaimah
RAK-AlMarjanIsland - Boolean - If the user typically rides in Al Marjan Island, Ras al Khaimah
RAK-DowntownRasAlKhaimah - Boolean - If the user typically rides in Downtown Ras Al Khaimah, Ras al Khaimah
RAK-JebelJais - Boolean - If the user typically rides in Jebel Jais, Ras al Khaimah
RAK-MinaAlArab - Boolean - If the user typically rides in Mina Al Arab, Ras al Khaimah

AlDhafra-Location - String - User text entry for where the user rides in Al Dhafra

UmmAlQuwain-Location - String - User text entry for where the user rides in Umm Al Quwain

Emirate-Location-Other - Boolean - If the user typically rides in another non-listed location
Emirate-Location-Other-Text - String - User text entry for where the user rides other than the listed locations

##########################################################################

In the '2025-06-16_Bike-UAE_Survey-Data-Analysis.py' you can find some data visulization and anaylsis.

##########################################################################