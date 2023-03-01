clear
clc
heart_space = 20;
delay = 350;
sd_len = 760;
SD_len = 14600;
x2 = linspace(0,1,sd_len);
X2 = linspace(0,1,SD_len);
pathname = 'E:\BG\new database';
path1 = 'E:\BG\ECG\';
path2 = 'E:\BG\PPG\';
path3 = 'E:\BG\ECG_PPG\';

img_path_list = dir(pathname);
file_num = length(img_path_list);
for f = 3:file_num
    data_file = dir(strcat(pathname,'\',img_path_list(f).name, '\*.mat'));
    ECG_num = length(data_file);
    label_file = strcat('E:\BG\','label\',img_path_list(f).name,'.txt');
    glucose = load(label_file);
    RRI_file = strcat('E:\BG\','RRI\',img_path_list(f).name,'.xlsx');
    RRI = xlsread(RRI_file);
        
    for F = 1:ECG_num
         file_name = strcat(pathname,'\',img_path_list(f).name,'\',data_file(F).name);         
         temp_Data = load(file_name);
         ECG = temp_Data.E_P(:,1);
         PPG = temp_Data.E_P(:,2);
        
         temp_glucose = glucose(:,F);
         temp_RRI = RRI(:,F);
         temp_rri = rmmissing(temp_RRI);
         len1 = length(temp_rri);
         all_temp = 0;
         for t = 2:len1
             dif_rri =  temp_rri(t,1)-temp_rri(t-1,1);
             all_temp = all_temp + dif_rri;             
         end
         aver_RRI = all_temp / (len1 - 1)*1000;
         P1 = floor(aver_RRI * 0.358);
         P2 = floor(aver_RRI * 0.644);
         
         total_num = floor(len1 / heart_space);
         len2 = total_num * heart_space;
         ECG_result = zeros(len2,sd_len);
         PPG_result = zeros(len2,sd_len);
         label_result = zeros(len2,1);
         x1 = linspace(0,1,P1+P2+1);
         N = 1;
         M = len2;
         for m = 1:len2
             po_ECG = floor(temp_rri(m,1)*1000);
             if temp_rri(m,1) <= 9300
                 ECG1 = po_ECG - P1;
                 ECG2 = po_ECG + P2;
                 temp_ecg = ECG(ECG1:ECG2,1);
                 temp_ECG = interp1(x1,temp_ecg,x2,'pchip');
                 temp_ppg = PPG(ECG1+delay:ECG2+delay,1);
                 temp_PPG = interp1(x1,temp_ppg,x2,'pchip');
                 ECG_result(m,:) = temp_ECG;
                 PPG_result(m,:) = temp_PPG;
                 if temp_rri(m,1) <= 300*N && temp_rri(m,1) <= 9300
                     label_result(m,1) = temp_glucose(N,1);
                 else
                     label_result(m,1) = temp_glucose(N+1,1);
                     N = N + 1;
                 end
             else
                 M = m;
                 break
             end
         end
        save_ECG = [ECG_result label_result];
        save_PPG = [PPG_result label_result];
        if M < len2
            save_ECG(M:len2,:) = [];
            save_PPG(M:len2,:) = [];
        end
         name_len = length(data_file(F).name)-4;
         new_name = data_file(F).name(1:name_len);
        save(strcat(path1,'\', new_name,'ECG.mat'), 'save_ECG');
        save(strcat(path2,'\', new_name,'PPG.mat'), 'save_PPG');
        
        W = 1;
        U = 1;
        Label_EP = zeros(total_num,1);
        SM = zeros(total_num,16);
        SC = zeros(total_num,16);
        Ku = zeros(total_num,16);
        Sk = zeros(total_num,16);
        KE = zeros(total_num,16);
        SE = zeros(total_num,16);
        PSE = zeros(total_num,16);
        C0 = zeros(total_num,16);
        CD = zeros(total_num,16);
        FD = zeros(total_num,16);
        
        for w = 0:heart_space:(len2 - heart_space)
            if w == 0
                q1 = 1;
                q2 = 21;
                Q1 = floor(temp_rri(q1,1)*1000);
                Q2 = floor(temp_rri(q2,1)*1000);
                mlECG = ECG(Q1:Q2,1);
                mlPPG = PPG(Q1:Q2,1);
                Label_EP(W,1) = temp_glucose(U,1);
                W = W + 1;
            else
                q1 = w ;
                q2 = w + heart_space;
                Q1 = floor(temp_rri(q1,1)*1000);
                Q2 = floor(temp_rri(q2,1)*1000);
                mlECG = ECG(Q1:Q2,1);
                mlPPG = PPG(Q1:Q2,1);
                if  temp_rri(w,1) <= 300*U && temp_rri(w,1) <= 9300
                     Label_EP(W,1) = temp_glucose(U,1);
                     W = W + 1;
                else
                    Label_EP(W,1) = temp_glucose(U + 1,1);
                    W = W + 1;
                    U = U + 1;
                end
            end
            X1 = linspace(0,1,length(mlECG));
            MLECG = interp1(X1,mlECG,X2,'pchip');
            MLPPG = interp1(X1,mlPPG,X2,'pchip');
            
            [c0,l0] = wavedec(MLECG,10,'db4');
            filter = MLECG - wrcoef('a',c0,l0,'db4',10);
            [c,l]=wavedec(filter,7,'db4');
            a7 = wrcoef('d',c,l,'db4',7);
            a6 = wrcoef('d',c,l,'db4',6);
            a5 = wrcoef('d',c,l,'db4',5);
            a4 = wrcoef('d',c,l,'db4',4);
            a3 = wrcoef('d',c,l,'db4',3);
            a2 = wrcoef('d',c,l,'db4',2);
            a1 = wrcoef('d',c,l,'db4',1);   
            com_ECG = [filter; a1; a2; a3; a4; a5; a6; a7];    

            [c0,l0] = wavedec(MLPPG,10,'db4');
            filter = MLPPG - wrcoef('a',c0,l0,'db4',10);
            [c,l]=wavedec(filter,7,'db4');
            a7 = wrcoef('d',c,l,'db4',7);
            a6 = wrcoef('d',c,l,'db4',6);
            a5 = wrcoef('d',c,l,'db4',5);
            a4 = wrcoef('d',c,l,'db4',4);
            a3 = wrcoef('d',c,l,'db4',3);
            a2 = wrcoef('d',c,l,'db4',2);
            a1 = wrcoef('d',c,l,'db4',1);   
            com_PPG = [filter; a1; a2; a3; a4; a5; a6; a7];  
            
            ECG_PPG_temp = [com_ECG; com_PPG];            
            R = size(ECG_PPG_temp,1);
            C = size(ECG_PPG_temp,2);
            ECG_PPG = zeros(R,C);
            for y = 1:R
                ECG_PPG(y,:) = mapminmax(ECG_PPG_temp(y,:), 0, 1);
            end
            
            for wav_data_i=1:R
                s = ECG_PPG(wav_data_i,:);
                Fs = 1000;
                T=1 / Fs;
                s = s';
                s_squr = s.^2;
              %% Signal mobility 
                for a=1:C-15
                    d(a) = s(a+15)-s(a);
                end
                for b=1:C-30
                    g(b) = d(b+15)-d(b);
                end           
               d_squr = d.^2;
               g_squr = g.^2;
               S0 = sqrt(sum(s_squr)/C);
               S1 = sqrt(sum(d_squr)/(C-15));
               S2 = sqrt(sum(g_squr)/(C-30));
               SM(W-1,wav_data_i) =S1/S0;   
               d = [];
               g = [];
             %% Signal complexity 
               SC(W-1,wav_data_i) = sqrt(S2^2/S1^2-S1^2/S0^2);   
           
             %% Kurtosis 
               Ku(W-1,wav_data_i) = kurtosis(s);
           
             %% Skewness 
               Sk(W-1,wav_data_i) = skewness(s);      
             %% Kolmogorov entropy
               y=s;
               duan =10;
               s_min=min(s);
               s_max=max(s);
               maxf(1)=abs(s_max-s_min);
               maxf(2)=s_min;
               duan_t=1.0/duan;
               jiange=maxf(1)*duan_t;
               pnum(1)=length(find(y<maxf(2)+jiange));
               for q=2:duan-1
                     pnum(q)=length(find((y>=maxf(2)+(q-1)*jiange)&(y<maxf(2)+q*jiange)));
               end
               pnum(duan)=length(find(y>=maxf(2)+(duan-1)*jiange));
               ppnum=pnum/sum(pnum);

               for r=1:duan
                   if ppnum(r)==0
                       Hi=0;
                   else
                      Hi(r)=-ppnum(r)*log2(ppnum(r));
                   end
               end
               len_Hi=length(Hi);
               for a=1:len_Hi-1
                   Inforloss(a)=Hi(a+1)-Hi(a);
               end
               KE(W-1,wav_data_i)=(sum(Inforloss))/len_Hi;    
               pnum = [];
               Hi = [];
               Inforloss = [];
               
             %% SE
               signal= s;
               se_total = 0;
               for nbins=2:200
                   [hdat1,h1] = hist(signal,nbins);
                    hdat1 = hdat1./sum(hdat1);
                    se_temp = -sum(hdat1.*log2(hdat1+eps));
                    se_total = se_temp + se_total;
               end 
               SE(W-1,wav_data_i) = se_total/198;  
               
               %%  Power spectral entropy
               [pxx, fpow] = pwelch(s, [], [], [], Fs); 
               % window =hanning(256);
               % Pxx=psd(CH10_all,256,Fs,window,128,'none');
               Px_x=10*log10(pxx);
               c=length(Px_x);
               sumPx =sum(Px_x);
               for d=1:c
                 Px(d)=Px_x(d)/sumPx;
               end
              for e =1:c
                Hw(d) =Px(e)*log2(Px(e));
              end
              PSE(W-1,wav_data_i)=-abs(sum(Hw));
              Px = [];
              Hw = [];
               
                   %% C0-complexity 
               s_four =fft(s);
               len_s_four = SD_len;
                % s_ifft=ifft(s_four);
               M=(sum(s_four.^2))/len_s_four;
               % [pxx, fpow] = pwelch(CH10_all, [], [], [], Fs); 
               for u=1:len_s_four
                    squr_s_four(u)=abs(s_four(u))^2;
                    if squr_s_four(u)> M
                        Y(u) =s_four(u);
                    else 
                        Y(u) =0;
                    end
               end
               y=ifft(Y);
               for g=1:len_s_four
                   abs_xy(g)=abs(s(g)-y(g));
               end
              C0_a1=sum(abs_xy.^2);
              C0_a0=sum(s.^2);
              C0(W-1,wav_data_i)= C0_a1/C0_a0;
              squr_s_four = [];
              Y = [];
              abs_xy = [];
              
              
    %%  Correlation dimension 
            mean_x =mean(s(:,1));
            A=[1:SD_len];
            num_data=0;
            n=0;
          for l=1:SD_len
               if l==1&&abs(s(l)-mean_x)<=0.01*mean_x
                num_data(l)=1;
      %         num_data(i)=sum(num_data);
                n=n+1;
               end
               if l~=1&&abs(s(l)-mean_x)<=0.01*mean_x
                 num_data(l)=num_data(l-1)+1;
       %         num_data(i)=sum(num_data);
                 n=n+1;
               end
               if n==0
                  num_data(l)=0;
               end
               if l~=1&&abs(s(l)-mean_x)>0.01*mean_x
                   num_data(l)=num_data(l-1);
               end  
          end
          p=polyfit(A,num_data,1);
         CD(W-1,wav_data_i)=p(1,1);             
         num_data = []; 
        %% Fractal dimension 
          serie = s';
          % Kmax =300000;
          Kmax= 20; 
        % X = NaN(Kmax,Kmax,N);
           for k = 1:Kmax
              for m = 1:k
                   limit = floor((SD_len-m)/k);
                   j = 1;
                   for im = m:k:(m + (limit*k))
                       X(k,m,j) = serie(im);
                       j = j + 1;
                   end  
              end
           end
          % Computing the length of each sub-serie:
          % L = NaN(1, Kmax);
           for m = 1:k
             R  = (SD_len - 1)/(floor((SD_len - m)/k) * k);
             aux= squeeze(X(k,m,logical(~isnan(X(k,m,:))))); 
             aux =aux(1:j,:);
             for i1 = 1:(length(aux) - 1)
                L_m1(i1) = abs(aux(i1+1) - aux(i1));
             end
             L_m2(m)= (sum(L_m1) * R)/k;
           end
           x_label = 1./(1:Kmax);
           moilcom = polyfit(log10(x_label),log10(L_m2),1);
           FD(W-1,wav_data_i) = moilcom(1,1);  
           X = [];
           L_m1 = [];
           L_m2 = [];
            end
                                                              
        end
        feature_temp = [SM SC Ku Sk KE SE PSE C0 CD FD Label_EP]; 
        save(strcat(path3,'\', new_name,'ECG_PPG_feature.mat'), 'feature_temp');

    end
end


