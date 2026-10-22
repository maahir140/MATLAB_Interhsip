function rainfall_predictor_app()
% RAINFALL_PREDICTOR_APP  -  Monsoon (Jun-Sep) rainfall predictor with GUI
%
%   * Ek hi file, koi API / server / backend / toolbox nahi (base MATLAB R2019b+).
%   * Data: agar CSV mili to real data, warna random synthetic data (demo mode).
%       - rainfall_data_cleaned.csv   (ya)   rainfall_in_india_1901-2015.csv
%   * Run:   >> rainfall_predictor_app
%
%   TASK (honest, leakage-free):
%     Predict this year's monsoon rainfall (JUN+JUL+AUG+SEP) using
%       JAN..MAY rainfall of the same year
%       + previous year's annual rainfall
%       + previous year's monsoon rainfall
%       + subdivision (one-hot)
%     Train: 1901-2000   |   Test: 2001-2015 (time-based split, no random shuffle)
%
%   MODELS: Baseline (subdivision average), Linear Regression, Ridge, kNN

    %% ================= 1. DATA =================
    months = {'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'};
    [T, dataSource] = loadData(months);
    T = sortrows(T, {'SUBDIVISION','YEAR'});

    subNames = unique(T.SUBDIVISION);          % string column
    nSub     = numel(subNames);
    [~, sid] = ismember(T.SUBDIVISION, subNames);
    yr       = T.YEAR;
    M        = T{:, months};
    n        = height(T);
    annual   = sum(M, 2);                      % annual = sum of 12 months
    mon      = sum(M(:, 6:9), 2);              % monsoon target (Jun-Sep)

    %% ================= 2. FEATURES (no leakage) =================
    prevAnn = nan(n,1);  prevMon = nan(n,1);
    for i = 2:n
        if sid(i) == sid(i-1) && yr(i) == yr(i-1) + 1
            prevAnn(i) = annual(i-1);
            prevMon(i) = mon(i-1);
        end
    end
    numF    = [M(:,1:5), prevAnn, prevMon];    % 7 numeric features
    valid   = ~any(isnan(numF), 2);
    isTrain = valid & yr <= 2000;
    isTest  = valid & yr >  2000;

    mu = mean(numF(isTrain,:), 1);
    sd = std(numF(isTrain,:), 0, 1);
    sd(sd == 0) = 1;
    Z   = (numF - mu) ./ sd;
    Dum = double(sid == (2:nSub));             % one-hot, first subdivision = reference
    X   = [Z, Dum];
    y   = mon;

    Xtr = X(isTrain,:);  ytr = y(isTrain);  str = sid(isTrain);
    Xte = X(isTest,:);   yte = y(isTest);   ste = sid(isTest);

    %% ================= 3. TRAIN MODELS =================
    modelNames = {'Baseline (subdivision avg)','Linear regression','Ridge regression','kNN (k=15)'};
    kNN = 15;

    subMean = zeros(nSub,1);
    for s = 1:nSub
        v = ytr(str == s);
        if isempty(v), subMean(s) = mean(ytr); else, subMean(s) = mean(v); end
    end

    A      = [ones(size(Xtr,1),1), Xtr];
    thOLS  = pinv(A) * ytr;                                    % least squares
    lam    = 10;
    R      = lam * eye(size(A,2));  R(1,1) = 0;
    thRdg  = (A'*A + R) \ (A'*ytr);                            % ridge

    nM    = numel(modelNames);
    preds = cell(1, nM);
    rmseV = zeros(1,nM);  maeV = zeros(1,nM);  r2V = zeros(1,nM);
    for m = 1:nM
        p = max(predictRows(m, Xte, ste), 0);
        preds{m} = p;
        e = yte - p;
        rmseV(m) = sqrt(mean(e.^2));
        maeV(m)  = mean(abs(e));
        r2V(m)   = 1 - sum(e.^2) / sum((yte - mean(yte)).^2);
    end
    gainV = 100 * (1 - rmseV ./ rmseV(1));                     % % better than baseline
    [~, bestM] = min(rmseV);
    [~, bestML] = min(rmseV(2:end));  defaultModel = bestML + 1;   % best real ML model

    metricsTbl = table(modelNames', round(rmseV',1), round(maeV',1), round(r2V',3), round(gainV',1), ...
        'VariableNames', {'Model','RMSE_mm','MAE_mm','R2','Gain_vs_Baseline_pct'});

    %% ================= 4. UI =================
    fig = uifigure('Name','Rainfall Predictor (India) - Monsoon Forecast', ...
                   'Position',[60 40 1100 660]);
    g0 = uigridlayout(fig,[1 1]);  g0.Padding = [0 0 0 0];
    tg = uitabgroup(g0);
    tab1 = uitab(tg,'Title','1. Predict');
    tab2 = uitab(tg,'Title','2. Model comparison');
    tab3 = uitab(tg,'Title','3. Data explorer');
    tab4 = uitab(tg,'Title','4. About');

    % ---------- TAB 1 : PREDICT ----------
    g1 = uigridlayout(tab1,[1 2]);
    g1.ColumnWidth = {360,'1x'};

    left = uipanel(g1,'Title','Input','FontWeight','bold');
    lg = uigridlayout(left,[8 1]);
    lg.RowHeight = {22,28,22,28,22,236,36,'1x'};

    uilabel(lg,'Text','Subdivision (auto-fills historical average)');
    dd = uidropdown(lg,'Items',cellstr(subNames),'ValueChangedFcn',@onSubdivision);

    uilabel(lg,'Text','Model');
    ddModel = uidropdown(lg,'Items',modelNames,'Value',modelNames{defaultModel});

    uilabel(lg,'Text','Inputs (mm)','FontWeight','bold');
    ig = uigridlayout(lg,[7 2]);
    ig.Padding = [0 0 0 0];
    ig.RowHeight = repmat({28},1,7);
    ig.ColumnWidth = {170,'1x'};
    inLabels = {'JAN rainfall','FEB rainfall','MAR rainfall','APR rainfall','MAY rainfall', ...
                'Prev. year ANNUAL','Prev. year Jun-Sep'};
    fields = gobjects(1,7);
    for k = 1:7
        uilabel(ig,'Text',inLabels{k});
        fields(k) = uieditfield(ig,'numeric','Value',0,'Limits',[0 Inf],'ValueDisplayFormat','%.1f');
    end

    bg = uigridlayout(lg,[1 3]);
    bg.Padding = [0 0 0 0];
    uibutton(bg,'Text','Predict','FontWeight','bold','BackgroundColor',[0.15 0.55 0.8], ...
             'FontColor','w','ButtonPushedFcn',@onPredict);
    uibutton(bg,'Text','Random input','ButtonPushedFcn',@onRandom);
    uibutton(bg,'Text','Clear','ButtonPushedFcn',@onClear);

    resultLbl = uilabel(lg,'Text','Predicted monsoon (Jun-Sep): --','FontSize',16, ...
        'FontWeight','bold','WordWrap','on','HorizontalAlignment','center');

    rg = uigridlayout(g1,[2 1]);
    rg.RowHeight = {'1x','1x'};
    ax1 = uiaxes(rg);
    title(ax1,'Historical monsoon rainfall of selected subdivision'); grid(ax1,'on');
    xlabel(ax1,'Jun-Sep rainfall (mm)'); ylabel(ax1,'Number of years');
    ax2 = uiaxes(rg);
    title(ax2,'Your Jan-May input'); grid(ax2,'on');
    xlabel(ax2,'Month'); ylabel(ax2,'Rainfall (mm)');

    % ---------- TAB 2 : MODEL COMPARISON ----------
    g2 = uigridlayout(tab2,[3 2]);
    g2.RowHeight = {'1x',150,50};
    axR = uiaxes(g2);
    bar(axR, 1:nM, rmseV, 'FaceColor',[0.2 0.55 0.8]);
    axR.XTick = 1:nM;
    axR.XTickLabel = {'Baseline','Linear','Ridge','kNN'};
    title(axR,'Test RMSE (lower is better)'); ylabel(axR,'RMSE (mm)'); grid(axR,'on');

    axS = uiaxes(g2);
    scatter(axS, yte, preds{bestM}, 12, 'filled'); hold(axS,'on');
    lim = [min(yte) max(yte)];
    plot(axS, lim, lim, 'r--', 'LineWidth', 1.5); hold(axS,'off');
    title(axS, ['Actual vs Predicted - ' modelNames{bestM} ' (test 2001-2015)']);
    xlabel(axS,'Actual (mm)'); ylabel(axS,'Predicted (mm)'); grid(axS,'on');

    tbl = uitable(g2,'Data',metricsTbl);
    tbl.Layout.Row = 2;  tbl.Layout.Column = [1 2];

    note = uilabel(g2,'WordWrap','on', ...
        'Text',sprintf(['Train rows = %d, Test rows = %d.  Baseline = "har subdivision ka average monsoon". ' ...
        'Agar ML model baseline se zyada better nahi hai, to iska matlab hai ki saal-dar-saal monsoon ' ...
        'in features se mushkil se predict hota hai - ye data ki limit hai, bug nahi.'], ...
        nnz(isTrain), nnz(isTest)));
    note.Layout.Row = 3;  note.Layout.Column = [1 2];

    % ---------- TAB 3 : DATA EXPLORER ----------
    g3 = uigridlayout(tab3,[2 1]);
    g3.RowHeight = {30,'1x'};
    ddX = uidropdown(g3,'Items',[{'All India (average)'}; cellstr(subNames)], ...
                     'ValueChangedFcn',@(~,~) updateExplorer());
    axE = uiaxes(g3);

    % ---------- TAB 4 : ABOUT ----------
    g4 = uigridlayout(tab4,[1 1]);
    uitextarea(g4,'Editable','off','Value',{ ...
        'RAINFALL PREDICTOR - ABOUT', ...
        '', ...
        ['Data source: ' dataSource], ...
        sprintf('Rows: %d | Subdivisions: %d | Years: %d-%d', n, nSub, min(yr), max(yr)), ...
        '', ...
        'Target: monsoon rainfall = JUN + JUL + AUG + SEP (mm)', ...
        'Features: JAN..MAY, previous-year annual, previous-year monsoon, subdivision', ...
        'Split: train 1901-2000, test 2001-2015 (time-based)', ...
        '', ...
        'Why not predict ANNUAL from 12 months? Because ANNUAL is almost the sum of those 12 months,', ...
        'so the model would just add numbers (R^2 ~ 0.998) and learn nothing - this is data leakage.', ...
        '', ...
        'No internet / API / backend is used. Everything runs locally inside MATLAB.'});

    updateExplorer();
    onSubdivision();

    %% ================= 5. CALLBACKS =================
    function onSubdivision(~,~)
        s = currentSub();
        rows = sid == s;
        v = mean(numF(rows,:), 1, 'omitnan');
        setFields(v);
    end

    function onPredict(~,~)
        vals = getFields();
        if all(vals == 0)
            uialert(fig,'Please enter at least one non-zero value.','No input');
            return;
        end
        s  = currentSub();
        m  = find(strcmp(modelNames, ddModel.Value));
        z  = (vals - mu) ./ sd;
        dm = zeros(1, nSub-1);
        if s > 1, dm(s-1) = 1; end
        p  = max(predictRows(m, [z, dm], s), 0);

        band = 1.28 * rmseV(m);                       % ~80% interval from test RMSE
        lo = max(p - band, 0);  hi = p + band;

        hist_ = ytr(str == s);
        p20 = pctl(hist_, 20);  p80 = pctl(hist_, 80);
        if p < p20
            cat = 'DEFICIENT monsoon';  col = [0.8 0.2 0.1];
        elseif p > p80
            cat = 'EXCESS monsoon';      col = [0.1 0.4 0.8];
        else
            cat = 'NORMAL monsoon';      col = [0.1 0.55 0.2];
        end
        resultLbl.Text = sprintf('Predicted monsoon (Jun-Sep):\n%.0f mm\n(80%% range: %.0f - %.0f mm)\n%s', ...
                                 p, lo, hi, cat);
        resultLbl.FontColor = col;

        cla(ax1);
        histogram(ax1, y(sid == s), 15, 'FaceColor',[0.6 0.75 0.9]);
        hold(ax1,'on');
        yl = ylim(ax1);
        plot(ax1,[p p],yl,'r-','LineWidth',2);
        plot(ax1,[lo lo],yl,'r:','LineWidth',1);
        plot(ax1,[hi hi],yl,'r:','LineWidth',1);
        hold(ax1,'off');
        title(ax1,['Historical monsoon: ' char(subNames(s)) '  (red line = prediction)']);

        cla(ax2);
        bar(ax2, 1:5, vals(1:5), 'FaceColor',[0.2 0.6 0.8]);
        ax2.XTick = 1:5;  ax2.XTickLabel = months(1:5);
        title(ax2,'Your Jan-May input'); ylabel(ax2,'Rainfall (mm)'); grid(ax2,'on');
    end

    function onRandom(~,~)
        % Random but realistic: sample around the subdivision's own history
        s = randi(nSub);
        dd.Value = char(subNames(s));
        rows = sid == s;
        m_ = mean(numF(rows,:), 1, 'omitnan');
        s_ = std(numF(rows,:), 0, 1, 'omitnan');
        v  = max(0, m_ + s_ .* randn(1,7));
        setFields(v);
        onPredict();
    end

    function onClear(~,~)
        setFields(zeros(1,7));
        resultLbl.Text = 'Predicted monsoon (Jun-Sep): --';
        resultLbl.FontColor = [0 0 0];
        cla(ax1);  cla(ax2);
    end

    function updateExplorer()
        cla(axE);
        if strcmp(ddX.Value, 'All India (average)')
            [uy, ~, gi] = unique(yr);
            v = accumarray(gi, mon, [], @mean);
            nm = 'All India';
        else
            s = find(strcmp(cellstr(subNames), ddX.Value), 1);
            uy = yr(sid == s);  v = mon(sid == s);  nm = ddX.Value;
        end
        plot(axE, uy, v, '-o', 'MarkerSize', 3); hold(axE,'on');
        plot(axE, uy, movmean(v,10), 'LineWidth', 2.5);
        pf = polyfit(uy, v, 1);
        plot(axE, uy, polyval(pf, uy), 'k--', 'LineWidth', 1.5);
        hold(axE,'off');
        legend(axE, 'Yearly monsoon', '10-year moving average', 'Linear trend', 'Location','best');
        title(axE, sprintf('%s - monsoon rainfall (trend = %+.1f mm/decade)', nm, pf(1)*10));
        xlabel(axE,'Year'); ylabel(axE,'Jun-Sep rainfall (mm)'); grid(axE,'on');
    end

    %% ================= 6. HELPERS (nested) =================
    function s = currentSub()
        s = find(strcmp(cellstr(subNames), dd.Value), 1);
    end

    function setFields(v)
        for q = 1:7, fields(q).Value = v(q); end
    end

    function v = getFields()
        v = zeros(1,7);
        for q = 1:7, v(q) = fields(q).Value; end
    end

    function p = predictRows(m, Xq, sq)
        switch m
            case 1
                p = subMean(sq);
            case 2
                p = [ones(size(Xq,1),1), Xq] * thOLS;
            case 3
                p = [ones(size(Xq,1),1), Xq] * thRdg;
            case 4
                w = 3;   % extra weight so neighbours come from the same subdivision
                Xk_tr = [Xtr(:,1:7), w*Xtr(:,8:end)];
                Xk_q  = [Xq(:,1:7),  w*Xq(:,8:end)];
                p = knnPredict(Xk_tr, ytr, Xk_q, kNN);
        end
        p = p(:);
    end
end

%% ================= LOCAL FUNCTIONS =================
function [T, src] = loadData(months)
    cands = {'rainfall_data_cleaned.csv','rainfall_in_india_1901-2015.csv','rainfall in india 1901-2015.csv'};
    f = '';
    for i = 1:numel(cands)
        if exist(cands{i},'file') == 2, f = cands{i}; break; end
    end
    if isempty(f)
        T = makeSyntheticData(months);
        src = 'DEMO MODE - random synthetic data (no CSV found in current folder)';
        return;
    end

    R = readtable(f, 'TreatAsMissing', {'NA','-',''});
    R.SUBDIVISION = string(R.SUBDIVISION);
    T = R(:, [{'SUBDIVISION','YEAR'}, months]);
    for j = 1:12
        c = T.(months{j});
        if ~isnumeric(c), c = str2double(string(c)); end
        T.(months{j}) = c;
    end

    % Missing values -> subdivision-wise mean, fallback overall mean
    subs = unique(T.SUBDIVISION);
    for j = 1:12
        c = T.(months{j});
        for s = 1:numel(subs)
            idx = T.SUBDIVISION == subs(s);
            m = mean(c(idx), 'omitnan');
            if isnan(m), m = mean(c, 'omitnan'); end
            c(idx & isnan(c)) = m;
        end
        T.(months{j}) = c;
    end

    % Remove duplicate (subdivision, year) rows
    [~, ia] = unique(T.SUBDIVISION + "_" + string(T.YEAR));
    T = T(sort(ia), :);
    src = ['REAL DATA - ' f];
end

function T = makeSyntheticData(months)
    % Fully random demo data: 8 regions x 115 years with monsoon-shaped months
    rng(7);
    shape = [1 1.5 2 4 8 40 90 80 50 15 5 2];  shape = shape / sum(shape);
    regions = ["North Plains","Hill Zone","Coastal West","East Delta", ...
               "Central Plateau","Desert West","Island Zone","North East"];
    years = (1901:2015)';
    nY = numel(years);
    A = zeros(0,12);  names = strings(0,1);  yrs = zeros(0,1);
    for r = 1:numel(regions)
        base = 500 + 500*rand();
        if r == 3 || r == 8, base = base * 2.5; end
        if r == 6, base = base * 0.3; end
        ann = base * exp(0.20 * randn(nY,1));           % random wet/dry years
        Mx  = ann .* shape .* exp(0.30 * randn(nY,12)); % random monthly noise
        A = [A; Mx];                                    %#ok<AGROW>
        names = [names; repmat(regions(r), nY, 1)];     %#ok<AGROW>
        yrs = [yrs; years];                             %#ok<AGROW>
    end
    T = table(names, yrs, 'VariableNames', {'SUBDIVISION','YEAR'});
    for j = 1:12, T.(months{j}) = A(:,j); end
end

function p = knnPredict(Xtr, ytr, Xq, k)
    p = zeros(size(Xq,1),1);
    for i = 1:size(Xq,1)
        d = sum((Xtr - Xq(i,:)).^2, 2);
        [~, ord] = sort(d);
        p(i) = mean(ytr(ord(1:min(k,numel(ord)))));
    end
end

function q = pctl(v, p)
    v = sort(v(:));  n = numel(v);
    pos = 1 + (n-1) * p / 100;
    lo = floor(pos);  hi = ceil(pos);
    q = v(lo) + (pos - lo) * (v(hi) - v(lo));
end
