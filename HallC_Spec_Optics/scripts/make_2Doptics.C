
/*
  This script reads an input file from optics data, and outputs 
  .pdf and/or 2D focal plane histogram objects. The data will be
  used to train the convolutional neural network model. 

*/

void make_2Doptics(int file_type=-1){
  
  if(file_type==1){
    cout << "Making 2D Optics Training Files . . ." << endl;
  }
  else if(file_type==2){
    cout << "Making 2D Optics Test Files . . ." << endl;
  }
  else{
    cout << "Please run the code using the argumnets (1) for training file or (2) for test files  " << endl;
    cout << "Example (from the command-line: root -l make_2Doptics(1) " << endl;
  }
  
  TString basename = "shms_pointtarg_7p5deg_2gev_wc_mscat_vac_shms_vary";
  TString inputroot;

  //Define output ROOTfilename
  TString outputhist;

  // Define quad tune variables (range, stepsize)
  float q1_min;  
  float q1_max;  
  float q1_step;   
	               
  float q2_min;  
  float q2_max;  
  float q2_step;    
	               
  float q3_min;  
  float q3_max;  
  float q3_step; 
  
  // Define training data Quad tuning minimum, maximum and stepsize
  // set (range, stepsize) based on the filenames range read in
  if(file_type==1){
    q1_min = 0.9;  
    q1_max = 1.12;  
    q1_step = 0.001;
    
    q2_min = 0.9;  
    q2_max = 1.12;  
    q2_step = 0.001;
    
    q3_min = 0.9;  
    q3_max = 1.12;  
    q3_step = 0.001;
  }

  // Define testing data Quad tuning minimum, maximum and stepsize
  else if(file_type==2){
    q1_min = 0.905;  
    q1_max = 1.12;  
    q1_step = 0.001;
    
    q2_min = 0.905;  
    q2_max = 1.12;  
    q2_step = 0.001;
    
    q3_min = 0.905;  
    q3_max = 1.12;  
    q3_step = 0.001;

  }

  // Define quads tuning of tests files (Q1 and Q3 are set to 1.00)
  
  
  gStyle->SetPalette(1,0);
  gStyle->SetOptStat(1);
  gStyle->SetOptFit(11);
  gStyle->SetTitleOffset(1.,"Y");
  gStyle->SetTitleOffset(.7,"X");
  gStyle->SetLabelSize(0.04,"XY");
  gStyle->SetTitleSize(0.06,"XY");
  gStyle->SetPadLeftMargin(0.12);

  
  // Begin Loop over quads tuning

  for(float q1=q1_min; q1<=(q1_max+q1_step); q1 = q1 + q1_step){
    
    for(float q2=q2_min; q2<=(q2_max+q2_step); q2 = q2 + q2_step){
      
      for(float q3=q3_min; q3<=(q3_max+q3_step); q3 = q3 + q3_step){

	//Define object array to store histogram objects
	TObjArray HList(0);

	if(file_type==1){
	  //Define input ROOT filename
	  inputroot =  "../ROOTfiles/training_files/" +basename + Form("_Q1_%.2f_Q2_%.2f_Q3_%.2f", q1, q2, q3) + ".root";
	}
	else if(file_type==2){
	  //Define input ROOT filename
	  inputroot =  "../ROOTfiles/testing_files/" +basename + Form("_Q1_%.2f_Q2_%.2f_Q3_%.3f", q1, q2, q3) + ".root";
	}
	
	//Check if filename exists
	Bool_t file_exists = !(gSystem->AccessPathName(inputroot));

	
	//If filename does not exist, move on to next 
	if(!file_exists) continue;

	if(file_type==1){
	  //Create output ROOTfile name to store the 2D optics plots to be done
	  outputhist = "../ROOTfiles/training_files/" + basename + Form("_Q1_%.3f_Q2_%.3f_Q3_%.3f_hist.root", q1, q2, q3);
	}
	else if(file_type==2){

	  //Create output ROOTfile name to store the 2D optics plots to be done
	  outputhist = "../ROOTfiles/testing_files/" + basename + Form("_Q1_%.3f_Q2_%.3f_Q3_%.3f_hist.root", q1, q2, q3);
	  cout << "(q1,q2,q3) = " << q1 << ", " << q2 << ", " << q3 << endl;
	  cout << "filename_in = " << inputroot << endl;
	  cout << "filename_out = " << outputhist << endl;

	}
	
	TString htitle=basename;
	TPaveLabel *title = new TPaveLabel(.15,.90,0.95,.99,htitle,"ndc");
	
	//Open Input ROOTfile, and get TTree
	TFile *fsimc = new TFile(inputroot); 
	TTree *tsimc = (TTree*) fsimc->Get("h1411");

	// Define branches
	Float_t         psxfp;    // position at focal plane ,+X is pointing down
	Float_t         psyfp;    // X x Y = Z so +Y pointing central ray left
	Float_t         psxpfp;   // dx/dz at focal plane
	Float_t         psypfp;   //  dy/dz at focal plane
	Float_t         psyptar;  //reconstructed
	Float_t         psyptari; //reconstructed
	Float_t         psytari;  //reconstructed
	Float_t         psytar;   //reconstructed
	Float_t         psxptar;  //reconstructed
	Float_t         ys;       //reconstructed
	Float_t         ys_calc;  //reconstructed
	Float_t         xs;       //reconstructed
	Float_t         xs_calc;  //reconstructed
	Float_t         delta;    //reconstructed
	Float_t         deltai;   //reconstructed
	Float_t         evtyp;    //reconstructed
	
	tsimc->SetBranchAddress("ysieve",&ys);
	tsimc->SetBranchAddress("xsieve",&xs);
	tsimc->SetBranchAddress("psdelta",&delta);
	tsimc->SetBranchAddress("psdeltai",&deltai);
	tsimc->SetBranchAddress("psxfp",&psxfp);
	tsimc->SetBranchAddress("psyfp",&psyfp);
	tsimc->SetBranchAddress("psxpfp",&psxpfp);
	tsimc->SetBranchAddress("psypfp",&psypfp);
	tsimc->SetBranchAddress("psytar",&psytar);
	tsimc->SetBranchAddress("psytari",&psytari);
	tsimc->SetBranchAddress("psyptar",&psyptar);
	tsimc->SetBranchAddress("psyptari",&psyptari);
	tsimc->SetBranchAddress("psxptar",&psxptar);
	tsimc->SetBranchAddress("evtype",&evtyp);
	
	Int_t type =1;

	int pixels = 200;
	
	TH2F *hxfp_yfp  = new TH2F("hxfp_yfp",    Form("Event type= %d ; Y_fp ; X_fp",type),     pixels, -20, 20,  pixels, -10, 10);
	HList.Add(hxfp_yfp);
	
	TH2F *hxfp_ypfp = new TH2F("hxfp_ypfp",   Form("Event type= %d ; Yp_fp ; X_fp",type),    pixels, -.1, .1,  pixels, -10, 10);
	HList.Add(hxfp_ypfp);
	
	TH2F *hxfp_xpfp = new TH2F("hxfp_xpfp",   Form("Event type= %d ; Xp_fp ; X_fp; ",type),  pixels, -.1, .1,  pixels, -10, 10);
	HList.Add(hxfp_xpfp);
	
	TH2F *hxpfp_yfp = new TH2F("hxpfp_yfp",   Form("Event type = %d ; Y_fp ; Xp_fp; ",type), pixels, -20, 20,  pixels, -.1, .1);
	HList.Add(hxpfp_yfp);
	
	TH2F *hxpfp_ypfp = new TH2F("hxpfp_ypfp", Form("Event type = %d ;  Yp_fp ; Xp_fp",type), pixels, -.1, .1,  pixels, -.1, .1);
	HList.Add(hxpfp_ypfp);
	
	TH2F *hypfp_yfp = new TH2F("hypfp_yfp",   Form("Event type = %d ; Y_fp ; Yp_fp",type),   pixels, -20, 20,  pixels, -.1, .1);
	HList.Add(hypfp_yfp);
	
	Long64_t nentries = tsimc->GetEntries();
	
	for (int i = 0; i < nentries; i++) {
	  
	  tsimc->GetEntry(i);
	  
	  if (evtyp == type) {
	    
	    hxfp_yfp   ->Fill(psyfp,  psxfp);
	    hxfp_ypfp  ->Fill(psypfp, psxfp);
	    hxfp_xpfp  ->Fill(psxpfp, psxfp);
	    hxpfp_ypfp ->Fill(psypfp, psxpfp);
	    hxpfp_yfp  ->Fill(psyfp,  psxpfp);
	    hypfp_yfp  ->Fill(psyfp,  psypfp);
	    
	  }
	} // end entry loop

	  
	TFile hsimc(outputhist,"recreate");
	HList.Write();

      } // end q3 loop

    } // end q2 loop

  } // end q1 loop
	
  //END Quads loop


}   
