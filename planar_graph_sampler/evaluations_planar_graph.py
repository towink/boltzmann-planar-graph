# Evals for N=10000, from maple and the old sage worksheet.
reference_evals_10000 = {
    'D(x*G_1_dx(x,y),y)': 1.09416749105326839436361139331519184699,
    'D_dx(x*G_1_dx(x,y),y)': 3.58480499936306184450275887839568999541,
    'D_dx_dx(x*G_1_dx(x,y),y)': 3844.44528283854975380346115822251416046,
    'Fusy_K': 0.00212262026119702814886982142258999866449,
    'Fusy_K_dx': 0.249953080767411054035497390586634583600,
    'Fusy_K_dx_dx': 840.103572875302590376914748297187794199,
    'Fusy_K_dy': 0.0214683058020723592469997145512349623450,
    'Fusy_K_dy_dy': 6.12142458745962758412596335033700990508,
    'Fusy_K_dx_dy': 72.0981198866882684638807996945396894562,
    'G(x,y)': 1.03814706567105803659247685817331208031,
    'G_1(x,y)': 0.0374374564718087765508715224183234143412,
    'G_1_dx(x,y)': 1.03982217197897390330627046797765213524,
    'G_1_dx_dx(x,y)': 1.1849448423679601844906264429882820439,
    'G_1_dx_dx_dx(x,y)': 26.97717307615521913829963604535756800,
    'G_2_dx(x*G_1_dx(x,y),y)': 0.039049710051328373527902391160990973285,
    'G_2_dx_dx(x*G_1_dx(x,y),y)': 1.05189847611928530057997563610421892748,
    'G_2_dx_dx_dx(x*G_1_dx(x,y),y)': 17.9872451737400462981579319960233386141,
    'G_3_arrow_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.0214683058020723592469997145512349623450,
    'G_dx(x,y)': 1.07948833665967802456253351678818766059,
    'G_dx_dx(x,y)': 2.35262291793778847713055101300644029098,
    'G_dx_dx_dx(x,y)': 33.0108508153309009525655605898420142252,
    'H(x*G_1_dx(x,y),y)': 0.00212262026119702814886982142258999866449,
    'H_dx(x*G_1_dx(x,y),y)': 0.326912770734535074726715514177226469305,
    'H_dx_dx(x*G_1_dx(x,y),y)': 1518.21806872915956879873505899143586685,

    # 'J_a(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.00106131013059851,
    'J_a(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.00424524052239404,
    # 'J_a_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.124976540383706,
    'J_a_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.499906161534824,
    'J_a_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1680.2071457506052,  # 2*Fusy_K_dx_dx

    'K(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.21670360435516503,
    'K_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 18.462984966367003,
    'K_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 70657.42195884386,
    'K_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1.998149518874907,
    'K_dy_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 6094.863636363636,
    'K_dy_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1189664441.662329,
    'K_snake(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1.536199432751157,
    'K_snake_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 6613.412698412699,
    'K_snake_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1301480185.0630813,
    'P(x*G_1_dx(x,y),y)': 0.0481585761872462889748092768486110858080,
    'P_dx(x*G_1_dx(x,y),y)': 1.87300065960603866766879075789499635792,
    'P_dx_dx(x*G_1_dx(x,y),y)': 2011.58863957504105608992077970919220810,
    'R_b(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.5749385151094553,
    'R_b_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1405.924242424242,
    'R_w(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 2.78591483387927,
    'R_w_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 4693.272727272727,
    'S(x*G_1_dx(x,y),y)': 0.0438862946048250772399322950439907625167,
    'S_dx(x*G_1_dx(x,y),y)': 1.38489156902248810210725260632346716819,
    'S_dx_dx(x*G_1_dx(x,y),y)': 314.638574534349128914805319521886085503,
    'x': 0.0367265761612091283824619433531762063743,
    'x*G_1_dx(x,y)': 0.0381891081932996814792832195422959507297,
    'y': 1.00000000000000
}

# N = 100
# J_a and J_a_dx are wrong.
reference_evals_100 = {
    'D(x*G_1_dx(x,y),y)': 1.09347848647472549211184239941769601061,
    'D_dx(x*G_1_dx(x,y),y)': 3.45102206434801157864803811245239712851,

    # 'Fusy_K': 0.00206469524549284585247141356302904687086,
    'G_3_arrow(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.00206469524549284585247141356302904687086,

    'Fusy_K_dx': 0.213391746841149010553294208298541737991,
    'Fusy_K_dy': 0.0182698213762051362133391897289610870537,
    'G_1(x,y)': 0.0372484305053690456202661877963311349072,
    'G_1_dx(x,y)': 1.03960692373287371278312170556092216601,
    'G_1_dx_dx(x,y)': 1.1831386535487874823112196643179832576,
    'G_2_dx(x*G_1_dx(x,y),y)': 0.038842683760013258631145523348425252874,
    'G_2_dx_dx(x*G_1_dx(x,y),y)': 1.05099440303963997980334319688998807042,
    'G_3_arrow_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.0182698213762051362133391897289610870537,
    'H(x*G_1_dx(x,y),y)': 0.00206469524549284585247141356302904687086,
    'H_dx(x*G_1_dx(x,y),y)': 0.276441303522129889594403984833856178203,
    'J_a(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.00103234762274642,
    'J_a_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.106695873420575,
    'K(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.21219385343653063,
    'K_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 15.48657625114519,
    'K_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 5592.386366298111,
    'K_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1.729773462783172,
    'K_dy_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 539.8558558558559,
    'K_dy_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1183895.1899970463,
    'K_snake(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1.254888507718696,
    'K_snake_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 543.8610354223433,
    'K_snake_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1277786.7614037835,
    'P(x*G_1_dx(x,y),y)': 0.0477986369539869322752735578478273809975,
    'P_dx(x*G_1_dx(x,y),y)': 1.80255895061461154228734452882283065547,
    'R_b(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.5120292887029289,
    'R_b_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 129.2273161413563,
    'R_w(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 2.577655310621243,
    'R_w_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 414.9509202453988,
    'S(x*G_1_dx(x,y),y)': 0.0436151542752457139840974280068395827400,
    'S_dx(x*G_1_dx(x,y),y)': 1.37202181021127014676628959879571029484,
    'x': 0.0365447705189290291920092340981152311040,
    'x*G_1_dx(x,y)': 0.0379921964577076229466965829549486315897,
    'y': 1.00000000000000}
my_evals_100 = {
    'x': 0.03654477051892902902890231687295889607237672073088,
    'y': 1.0,
    'x*G_1_dx(x,y)': 0.037992196457707622770077162176638051282469293180142,
    'G_1(x,y)': 0.037248430505369042655406418269040091926205146504296,
    'G_1_dx(x,y)': 1.039606923732857137361084533222518118179044019566,
    'G_1_dx_dx(x,y)': 1.1831386535482449173134658618702915140873062943159,
    'G_1_dx_dx_dx(x,y)': 8.2033647718450,
    # manually copied from maple because I can't evaluate it yet in the evals-notebook
    'D(x*G_1_dx(x,y),y)': 1.0934784864747254915023248813193805475499098636118,
    'S(x*G_1_dx(x,y),y)': 0.043615154275245713741771730592115881283209058946452,
    'P(x*G_1_dx(x,y),y)': 0.047798636953986931956906640071515346987277814746075,
    'H(x*G_1_dx(x,y),y)': 0.0020646952454928458036465106557493192794229899192356,
    'D_dx(x*G_1_dx(x,y),y)': 3.4510220643480115144243380233721683731619066447783,
    'S_dx(x*G_1_dx(x,y),y)': 1.3720218102112701396529130637046696972452061492002,
    'P_dx(x*G_1_dx(x,y),y)': 1.8025589506146115082616776900279594827115305731887,
    'H_dx(x*G_1_dx(x,y),y)': 0.27644130352212986650974726963953919320516992238931,
    'D_dx_dx(x*G_1_dx(x,y),y)': 363.62762263665562764249709805713077302248646202619,
    'S_dx_dx(x*G_1_dx(x,y),y)': 40.275166251505363143152291213855531754448702911982,
    'P_dx_dx(x*G_1_dx(x,y),y)': 192.64963438818684010346149119766549272453426798367,
    'H_dx_dx(x*G_1_dx(x,y),y)': 130.70282199696342439588331564560974854350349113054,

    'G_2_arrow(x*G_1_dx(x,y),y)': 1.0467392432373627460559211997088480053042396476753,
    'G_2_arrow_dx(x*G_1_dx(x,y),y)': 1.7255110321740057893240190562261985644524959515063,

    'G_2(x*G_1_dx(x,y),y)': 0.00073195292001645580388832824672663971527295817313461,
    'G_2_dx(x*G_1_dx(x,y),y)': 0.038842683760013258445519500642317738951803345086467,
    'G_2_dx_dx(x*G_1_dx(x,y),y)': 1.0509944030396399792319166014847185408878359517925,
    'G_2_dx_dx_dx(x*G_1_dx(x,y),y)': 3.2353553923297,
    # manually copied from maple because I can't evaluate it yet in the evals-notebook
    'G_3_arrow(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.0020646952454928458036465106557493192794229899192361,
    'G_3_arrow_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.018269821376205134717036849924201389091751404245262,
    'G_3_arrow_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.21339174684114899380576541145247204032635782258080,
    'G_3_arrow_dy_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.54409220326099450058270516465316396542717357195357,
    'G_3_arrow_dx_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 6.5942273240627609526630544096024451633805353355519,
    'G_3_arrow_dy_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 6.5942273240627609526630544096024451633805353355519,
    'G_3_arrow_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 72.065867209945589475406224907961859422425138334402,
    'J_a(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.0041293904909856916072930213114986385588459798384722,
    'J_a_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.42678349368229798761153082290494408065271564516161,
    'J_a_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 144.13173441989117895081244981592371884485027666880,
    'J(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.026445920425706887537828284367163398511121815482753,
    'J_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 2.6261958909828053330833267898475644233072494528005,
    'J_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 798.58988297408267577179452242344034707485520508767,
    'I(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.21219384716604521592438059123910203727388046615813,
    'I_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 15.486584147894425062357116434242264627767808265426,
    'I_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 5592.3862581715369999530260243534648619766888147153,
    'R_w(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 2.5776552900027291599002542759827997291057043800010,
    'R_b(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.51202931170089506810252223728268060219699338617808,
    'R_w_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 414.95092669997637875249245345223494577345960988135,
    'R_b_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 129.22731585966000761133668501626080242695835947592,
    'R_w_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 910684.27434137689807958461440929661434466705852769,
    'R_b_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 273211.02924871289865944117867346027770592697920272,
    'K(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.21219384716604521592438059123910203727388046615813,
    'K_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 15.486584147894425062357116434242264627767808265426,
    'K_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 5592.3862581715369999530260243534648619766888147153,
    'K_dy_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1183895.3035900897967390257930827568920505940377304}

my_evals_1000 = {
    'x': 0.03671004837554730320585565246738194483330135047639,
    'y': 1.0,
    'x*G_1_dx(x,y)': 0.038171203376643647123001266577519529427204851725851,
    'G_1(x,y)': 0.037420270675653892910168862532116674556872177618920,
    'G_1_dx(x,y)': 1.039802589910744775467874628412844342889448731901,
    'G_1_dx_dx(x,y)': 1.1846774415007057207792588268871939991318421157601,
    'D(x*G_1_dx(x,y),y)': 1.0941036513737400522118026669545430752530282427677,
    'S(x*G_1_dx(x,y),y)': 0.043861527521017791126774419147906144111042338618860,
    'P(x*G_1_dx(x,y),y)': 0.048125221487014818482817693410554460752663089611262,
    'H(x*G_1_dx(x,y),y)': 0.0021169023657074426022105543960824703893228145375372,
    'D_dx(x*G_1_dx(x,y),y)': 3.5519657293342095458792867418830730336055188098875,
    'S_dx(x*G_1_dx(x,y),y)': 1.3820896025616211482731394843261627974489057626773,
    'P_dx(x*G_1_dx(x,y),y)': 1.8557909831586291912451195890218377616706609608521,
    'H_dx(x*G_1_dx(x,y),y)': 0.31408514361395920636102766853507247448595208635800,
    'D_dx_dx(x*G_1_dx(x,y),y)': 1198.5261967251175565451537383311179386247597318630,
    'S_dx_dx(x*G_1_dx(x,y),y)': 106.45360990442147607854023691337237041564433894210,
    'P_dx_dx(x*G_1_dx(x,y),y)': 629.06945499533058568489298896805531253171571815682,
    'H_dx_dx(x*G_1_dx(x,y),y)': 463.00313182536549478172051244969025567739967476408,
    'G_2(x*G_1_dx(x,y),y)': 0.00073892287127617661459993599293425427960666030913784,
    'G_2_dx(x*G_1_dx(x,y),y)': 0.039030877742186946352245960937146683364532653234585,
    'G_2_dx_dx(x*G_1_dx(x,y),y)': 1.0517284581204708029034131990620553003959227304132,
    'G_3_arrow_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.020669113454789892666056470048883439625788429997974,
    'G_3_arrow_dy_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1.8832363425841084443775142012829405373722053241324,
    'G_3_arrow_dx_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 22.324260357449042550210845256009367648403842123460,
    'J_a(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.0042338047314148852044211087921649407786456290750725,
    'J_a_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.48133832193364980338830771963385636027526597107474,
    'J_a_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 511.76173086754009827551589064038874535497600612223,
    'R_w(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 2.7337737924641306947655530251139220056317170052092,
    'R_b(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.55930911997651496999368217399137136954871009338866,
    'R_w_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1440.0000144409987641269822629687575880307626575360,
    'R_b_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 435.46295256478111232833082490749823548162109630546,
    'R_w_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 28995383.496317448194900546389970488243614129784017,
    'R_b_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 8653655.0418790137393118506480717721565050247125358,
    'K(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.21625624230717231974541823454624842354767457972622,
    'K_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 17.687826318309949924696674564166789143310610595413,
    'K_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 21099.810055660399528441093406715351075871352344164,
    'K_dy_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 37649038.538196461934212397038042260400119154496553
}

my_evals_10000 = {
    'x': 0.036726576161209128382461943353176206374355475081909,
    'y': 1.0,
    'x*G_1_dx(x,y)': 0.038189108193299681479283219542295950729760843631217,
    'G_1(x,y)': 0.037437456471808779342268850532987352290319929325382,
    'G_1_dx(x,y)': 1.0398221719789738958184226082412932427610152768606,
    'G_1_dx_dx(x,y)': 1.1849448438620223820868768620583441284556572926034,
    'G_1_dx_dx_dx(x,y)': 26.977173076155,
    'D(x*G_1_dx(x,y),y)': 1.0941674910532683943636113933151918469894271590553,
    'S(x*G_1_dx(x,y),y)': 0.043886294604825077239932295043990762516803616618451,
    'P(x*G_1_dx(x,y),y)': 0.048158576187246288974809276848611085808112395269737,
    'H(x*G_1_dx(x,y),y)': 0.0021226202611970281488698214225899986645111471670745,
    'D_dx(x*G_1_dx(x,y),y)': 3.5848049993630618445027588783956899943193816543472,
    'S_dx(x*G_1_dx(x,y),y)': 1.3848915690224881021072526063234671681016605005273,
    'P_dx(x*G_1_dx(x,y),y)': 1.8730006596060386676687907578949963573467247623625,
    'H_dx(x*G_1_dx(x,y),y)': 0.32691277073453507472671551417722646887099639145742,
    'D_dx_dx(x*G_1_dx(x,y),y)': 3844.4452828385497538034611582225139898232474961366,
    'S_dx_dx(x*G_1_dx(x,y),y)': 314.63857453434912891480531952188607208435578194717,
    'P_dx_dx(x*G_1_dx(x,y),y)': 2011.5886395750410560899207797091921189459887918799,
    'H_dx_dx(x*G_1_dx(x,y),y)': 1518.2180687291595687987350589914357987929029223095,
    'G_2(x*G_1_dx(x,y),y)': 0.00073962188057625638648984729031898781819997789768603,
    'G_2_dx(x*G_1_dx(x,y),y)': 0.039049710051328373527902391160990973331066587592480,
    'G_2_dx_dx(x*G_1_dx(x,y),y)': 1.0518984761192853005799756361042239052000978707810,
    'G_2_dx_dx_dx(x*G_1_dx(x,y),y)': 17.9872451737400,
    'G_3_arrow(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.0021226202611970281488698214225899986645111471670744,
    'G_3_arrow_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.021468305802072359246999714551234962354122042533331,
    'G_3_arrow_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.24995308076741105403549739058663458315721462582360,
    'G_3_arrow_dy_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 6.1214245874596275841259633503370093899591380845857,
    'G_3_arrow_dx_dy(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 72.098119886688268463880799694539688516747423766278,
    'G_3_arrow_dy_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 72.098119886688268463880799694539688516747423766278,
    'G_3_arrow_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 840.10357287530259037691474829718774332472314230274,
    'J_a(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.0042452405223940562977396428451799973290222943341489,
    'J_a_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.49990616153482210807099478117326916631442925164721,
    'J_a_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1680.2071457506051807538294965943754866494462846055,
    'J(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.027165067487758979734968439496698244923953261991112,
    'J_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 3.0257713239280163253311383373161526262206620225241,
    'J_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 8978.5285235570588626747914623685863224098006299293,
    'I(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.21670364168399074747420924986339460290266818776345,
    'I_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 18.462969459803365888397738940951942650392705612949,
    'I_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 70657.409259074574910312830524839969026895138706317,
    'R_w(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 2.7859148613861976484300825888244909207509595329658,
    'R_b(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.57493851558207939193703168856805772168988374098287,
    'R_w_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 4693.2730608962529593998551522698997460050538242594,
    'R_b_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1405.9242020095371044076574013468011750337315315226,
    'R_w_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 916347175.24694309134967136839509420633930881049295,
    'R_b_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 273318149.32551582397773305813133784371226736244663,
    'K(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 0.21670364168399074747420924986339460290266818776345,
    'K_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 18.462969459803365888397738940951942650392705612949,
    'K_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 70657.409259074574910312830524839969026895138706317,
    'K_dy_dx_dx(x*G_1_dx(x,y),D(x*G_1_dx(x,y),y))': 1189665324.5724589153274044265264320500515761729396}
