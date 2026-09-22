using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

internal sealed class EvaluationWindow : Form
{
    private readonly WebView2 view = new WebView2();
    private readonly string root = AppDomain.CurrentDomain.BaseDirectory;
    private readonly string instance = Guid.NewGuid().ToString("N");
    private readonly string origin = "http://127.0.0.1:17893";
    private readonly string profile;
    private readonly bool smoke;
    private readonly StringBuilder diagnostics = new StringBuilder();
    private readonly JavaScriptSerializer json = new JavaScriptSerializer();
    private Process backend;
    private bool closing;
    private string downloaded;

    public EvaluationWindow(bool smokeTest)
    {
        smoke = smokeTest;
        profile = smoke ? Path.Combine(Path.GetTempPath(), "evaluation-webview-test-" + instance)
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "NeuroDiscovery-Human-Evaluation-WebView");
        Directory.CreateDirectory(profile);
        Text = "NeuroDiscovery Human Evaluation · WebView2";
        Width = 1280;
        Height = 900;
        MinimumSize = new System.Drawing.Size(800, 600);
        view.Dock = DockStyle.Fill;
        Controls.Add(view);
        if (smoke) { Opacity = 0; ShowInTaskbar = false; }
        Shown += async delegate { await Boot(); };
        FormClosing += delegate { closing = true; StopBackend(); view.Dispose(); };
    }

    private bool IsLocal(string address)
    {
        Uri target;
        return Uri.TryCreate(address, UriKind.Absolute, out target) && target.GetLeftPart(UriPartial.Authority) == origin;
    }

    private void StopBackend()
    {
        if (backend == null) return;
        try
        {
            if (!backend.HasExited)
            {
                backend.StandardInput.Close();
                if (!backend.WaitForExit(4000)) backend.Kill();
            }
        }
        catch (InvalidOperationException) { }
    }

    private async Task Boot()
    {
        try
        {
            CoreWebView2Environment.GetAvailableBrowserVersionString();
            var environment = await CoreWebView2Environment.CreateAsync(null, Path.Combine(profile, "browser"));
            if (closing) return;
            await view.EnsureCoreWebView2Async(environment);
            view.CoreWebView2.Settings.AreHostObjectsAllowed = false;
            view.CoreWebView2.Settings.IsWebMessageEnabled = false;
            view.CoreWebView2.Settings.AreDevToolsEnabled = smoke;
            view.CoreWebView2.PermissionRequested += delegate(object sender, CoreWebView2PermissionRequestedEventArgs args) { args.State = CoreWebView2PermissionState.Deny; };
            view.CoreWebView2.NavigationStarting += delegate(object sender, CoreWebView2NavigationStartingEventArgs args) { if (!IsLocal(args.Uri)) args.Cancel = true; };
            view.CoreWebView2.FrameNavigationStarting += delegate(object sender, CoreWebView2NavigationStartingEventArgs args) { if (args.Uri != "about:blank" && !IsLocal(args.Uri)) args.Cancel = true; };
            view.CoreWebView2.NewWindowRequested += delegate(object sender, CoreWebView2NewWindowRequestedEventArgs args)
            {
                args.Handled = true;
                Uri target;
                if (args.IsUserInitiated && Uri.TryCreate(args.Uri, UriKind.Absolute, out target) && target.Scheme == "https")
                    Process.Start(new ProcessStartInfo(target.AbsoluteUri) { UseShellExecute = true });
            };
            view.CoreWebView2.DownloadStarting += delegate(object sender, CoreWebView2DownloadStartingEventArgs args)
            {
                if (!args.DownloadOperation.Uri.StartsWith("blob:" + origin + "/", StringComparison.Ordinal)) { args.Cancel = true; return; }
                string destination;
                if (smoke) destination = Path.Combine(profile, "export.json");
                else
                {
                    using (var dialog = new SaveFileDialog())
                    {
                        dialog.Filter = "JSON files (*.json)|*.json";
                        dialog.FileName = Path.GetFileName(args.ResultFilePath);
                        if (dialog.ShowDialog(this) != DialogResult.OK) { args.Cancel = true; return; }
                        destination = dialog.FileName;
                    }
                }
                args.ResultFilePath = destination;
                args.Handled = true;
                var download = args.DownloadOperation;
                download.StateChanged += delegate { if (download.State == CoreWebView2DownloadState.Completed) downloaded = destination; };
            };
            var start = new ProcessStartInfo(Path.Combine(root, "runtime/python/python.exe"));
            start.Arguments = "-I -B \"" + Path.Combine(root, "evaluation-webview-server.py") + "\" --data \"" + Path.Combine(profile, "evaluation-data") + "\"";
            start.WorkingDirectory = root;
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.RedirectStandardInput = true;
            start.RedirectStandardError = true;
            start.EnvironmentVariables.Remove("PYTHONPATH");
            start.EnvironmentVariables.Remove("PYTHONHOME");
            start.EnvironmentVariables["NEURODISCOVERY_EVALUATION_INSTANCE"] = instance;
            backend = new Process { StartInfo = start, EnableRaisingEvents = true };
            backend.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs args) { if (args.Data != null) lock (diagnostics) { if (diagnostics.Length < 8000) diagnostics.AppendLine(args.Data); } };
            backend.Exited += delegate
            {
                if (!closing && IsHandleCreated) BeginInvoke((Action)delegate
                {
                    if (!closing) { if (!smoke) MessageBox.Show(this, "Evaluation server stopped. Saved answers remain on disk. Please reopen the application."); Environment.ExitCode = 1; Close(); }
                });
            };
            backend.Start();
            backend.BeginErrorReadLine();
            bool ready = false;
            for (int attempt = 0; attempt < 100 && !closing; attempt++)
            {
                if (backend.HasExited) throw new Exception("Backend failed: " + diagnostics.ToString());
                try
                {
                    var request = (HttpWebRequest)WebRequest.Create(origin + "/api/health");
                    request.Proxy = null;
                    request.Timeout = 500;
                    using (var response = await request.GetResponseAsync())
                    using (var reader = new StreamReader(response.GetResponseStream()))
                    {
                        var health = json.Deserialize<System.Collections.Generic.Dictionary<string, object>>(await reader.ReadToEndAsync());
                        ready = health.ContainsKey("instance_id") && (string)health["instance_id"] == instance;
                    }
                }
                catch (WebException) { }
                if (ready) break;
                await Task.Delay(200);
            }
            if (closing) return;
            if (!ready) throw new Exception("Local evaluation backend did not become ready. Port 17893 may be occupied.");
            view.CoreWebView2.Navigate(origin);
            if (smoke) { await Smoke(); Close(); }
        }
        catch (Exception error)
        {
            Environment.ExitCode = 1;
            File.WriteAllText(Path.Combine(profile, "startup-error.txt"), error.ToString());
            if (!closing && !smoke) MessageBox.Show(this, "Unable to start Human Evaluation.\nA Windows WebView2 Runtime is required. No runtime is installed automatically.\n" + error.Message, "Human Evaluation");
            Close();
        }
    }

    private async Task Evaluate(string script) { await view.CoreWebView2.ExecuteScriptAsync(script); }

    private async Task Until(string expression)
    {
        for (int attempt = 0; attempt < 100; attempt++)
        {
            if (await view.CoreWebView2.ExecuteScriptAsync("Boolean(" + expression + ")") == "true") return;
            await Task.Delay(100);
        }
        string snapshot = await view.CoreWebView2.ExecuteScriptAsync("JSON.stringify({url:location.href,body:document.body?.innerText,frame:document.querySelector('iframe')?.contentDocument?.body?.innerText})");
        throw new Exception("UI check timed out: " + expression + "\n" + snapshot);
    }

    private async Task Smoke()
    {
        await Until("document.querySelectorAll('[data-study]').length === 2");
        await Until("document.readyState === 'complete'");
        await Evaluate("document.querySelector('[data-study=discovery-study]').click()");
        string doc = "document.querySelector('iframe').contentDocument";
        await Until(doc + "?.querySelector('#assignment')?.options.length > 1");
        await Evaluate("(()=>{const doc=" + doc + ";doc.querySelector('[name=code]').value='TEST-WEBVIEW';doc.querySelector('[name=experience]').value='3-5';doc.querySelector('#assignment').value='P01';doc.querySelector('#setup-form').requestSubmit();})()");
        await Until(doc + ".querySelector('#workspace').hidden === false");
        await Evaluate(doc + ".querySelector('#pause').click()");
        await Until("document.querySelector('#home').hidden === false");
        await Evaluate("document.querySelector('[data-study=discovery-study]').click()");
        await Until(doc + "?.querySelector('#resume-banner')?.hidden === false");
        view.CoreWebView2.Navigate(origin + "/study?embedded=1");
        await Until("document.querySelector('#assignment')?.options.length > 1");
        await Until("document.querySelector('[data-view=results]').hidden");
        await Evaluate("document.querySelector('#participant-id').value='TEST-WEBVIEW';document.querySelector('#participant-experience').value='3-5';document.querySelector('#assignment').value='P01';document.querySelector('#setup-form').requestSubmit()");
        await Until("document.querySelector('#workbench-view').classList.contains('active')");
        await Evaluate("document.querySelector('#save-progress').click()");
        await Task.Delay(500);
        view.CoreWebView2.Reload();
        await Until("document.querySelector('#resume-session-btn')?.offsetParent");
        await Evaluate("EvaluationExport.exportResults({code:'TEST-WEBVIEW'});void 0");
        await Until("document.querySelector('dialog.evaluation-export-dialog[open]')");
        await Evaluate("document.querySelector('dialog.evaluation-export-dialog[open] .primary').click()");
        for (int attempt = 0; attempt < 100 && downloaded == null; attempt++) await Task.Delay(100);
        if (downloaded == null) throw new Exception("WebView JSON download did not finish");
        var export = json.Deserialize<System.Collections.Generic.Dictionary<string, object>>(File.ReadAllText(downloaded));
        if ((string)export["participant_code"] != "TEST-WEBVIEW") throw new Exception("Incorrect participant export");
        foreach (string key in new[] { "human_evaluation_1", "human_evaluation_2" })
        {
            var section = (System.Collections.Generic.Dictionary<string, object>)export[key];
            if ((string)section["status"] == "not_included") throw new Exception("Missing exported evaluation: " + key);
        }
        File.WriteAllText(Path.Combine(profile, "SMOKE_PASSED.json"), json.Serialize(new { status = "passed", he1_save_resume = true, he2_save_resume = true, webview_json_download = true, backend_pid = backend.Id, profile = profile }));
    }

    [STAThread]
    public static int Main(string[] arguments)
    {
        bool smokeTest = Array.IndexOf(arguments, "--smoke-test") >= 0;
        bool created;
        using (var mutex = new Mutex(true, "Local\\NeuroDiscovery-Human-Evaluation-WebView-" + Environment.UserName, out created))
        {
            if (!created) { if (!smokeTest) MessageBox.Show("Human Evaluation WebView is already open."); return 1; }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new EvaluationWindow(smokeTest));
            mutex.ReleaseMutex();
        }
        return Environment.ExitCode;
    }
}
