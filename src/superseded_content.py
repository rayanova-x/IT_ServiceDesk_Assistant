# Old/superseded versions of 25 existing subcategories. Each references the SAME
# subcategory as an existing current SOP, but with an earlier date and guidance that
# is outdated or partially conflicting -- a realistic "never cleaned up the KB" case.

SUPERSEDED = [
    {
        "subcategory": "Access denied by server",
        "category": "Access",
        "old_date": "2024-02-10",
        "old_symptoms": "User cannot log into the legacy internal portal (pre-SSO migration).",
        "old_resolution": [
            "Have the user try the old standalone login page at portal-legacy.internal instead of the SSO redirect.",
            "If that fails, manually reset their local portal password via the legacy admin console (deprecated as of the 2025 SSO migration).",
        ],
        "note": "This procedure predates the single sign-on migration completed in 2025. The legacy admin console referenced here has since been decommissioned.",
    },
    {
        "subcategory": "Access control list misconfigured",
        "category": "Access",
        "old_date": "2024-04-02",
        "old_symptoms": "User cannot access a shared drive folder.",
        "old_resolution": [
            "Manually edit NTFS permissions directly on the file server via Remote Desktop.",
            "Add the individual user account directly to the folder ACL.",
        ],
        "note": "Direct NTFS edits and individual-user ACL entries are no longer the standard approach -- current process uses group-based access requests through the access management system.",
    },
    {
        "subcategory": "Access denied by network policy",
        "category": "Access",
        "old_date": "2023-12-01",
        "old_symptoms": "User blocked from an internal tool while on the corporate network.",
        "old_resolution": [
            "Ask the user to connect via the guest Wi-Fi network as a temporary workaround.",
            "Submit a manual firewall exception request via email to the network administrator.",
        ],
        "note": "Guest Wi-Fi is not an approved workaround for internal tool access under current security policy, and firewall exceptions are now submitted via the ticketing system, not email.",
    },
    {
        "subcategory": "Unauthorized login attempt",
        "category": "Access",
        "old_date": "2024-01-20",
        "old_symptoms": "User reports an unrecognized login notification.",
        "old_resolution": [
            "Advise the user to simply ignore the notification if their account still works normally.",
            "No password reset needed unless the user explicitly reports being locked out.",
        ],
        "note": "This guidance predates the current mandatory-reset policy for any unauthorized login notification, regardless of whether the account still functions.",
    },
    {
        "subcategory": "Access denied by BYOD policy",
        "category": "Access",
        "old_date": "2024-06-15",
        "old_symptoms": "Personal device blocked from the corporate network.",
        "old_resolution": [
            "Direct the user to install the legacy MDM agent (v2) from the shared drive.",
            "BYOD is only supported for iOS devices under this policy version.",
        ],
        "note": "The v2 MDM agent has been replaced, and BYOD support was extended to Android devices in the 2025 policy update -- this article was not updated to reflect that.",
    },
    {
        "subcategory": "Suspicious cookie theft",
        "category": "Security",
        "old_date": "2024-03-05",
        "old_symptoms": "Browser session flagged for possible cookie hijacking.",
        "old_resolution": [
            "Advise the user to clear cookies and continue working; no session invalidation required.",
            "Log the incident for monthly security review only.",
        ],
        "note": "Current policy requires immediate session invalidation for any confirmed cookie theft alert, not a monthly batch review -- this reflects an older, less strict incident response tier.",
    },
    {
        "subcategory": "Phishing website blocked",
        "category": "Security",
        "old_date": "2024-05-18",
        "old_symptoms": "User clicked a phishing link; web filter blocked the destination.",
        "old_resolution": [
            "No action needed if the site was successfully blocked before page load.",
            "Report to Security Team only if the user entered credentials.",
        ],
        "note": "Current guidance requires reporting every blocked phishing click org-wide for campaign tracking, even when the block was successful and no credentials were entered.",
    },
    {
        "subcategory": "Suspicious contractor compromise",
        "category": "Security",
        "old_date": "2023-11-22",
        "old_symptoms": "Unusual activity on a contractor account.",
        "old_resolution": [
            "Email the contractor's manager and wait for a response before taking any account action.",
            "Do not suspend the account unless compromise is fully confirmed.",
        ],
        "note": "Current policy mandates immediate temporary suspension pending confirmation, rather than waiting for manager response first -- this older guidance introduces unacceptable delay under current standards.",
    },
    {
        "subcategory": "Suspicious login location",
        "category": "Security",
        "old_date": "2024-02-28",
        "old_symptoms": "Login flagged from an unusual geolocation.",
        "old_resolution": [
            "Send the user an automated email asking them to confirm; take no action until they respond.",
            "Sessions remain active while awaiting user confirmation.",
        ],
        "note": "Current guidance requires revoking the session immediately pending confirmation, not leaving it active -- this reflects an outdated, higher-risk-tolerance approach.",
    },
    {
        "subcategory": "Suspicious ICS alert",
        "category": "Security",
        "old_date": "2023-10-09",
        "old_symptoms": "Anomaly flagged on industrial control system monitoring.",
        "old_resolution": [
            "Log the alert and review it during the next scheduled weekly OT security meeting.",
            "No immediate isolation required unless a device fails outright.",
        ],
        "note": "This predates the current same-day isolation requirement for any unexplained ICS alert, introduced after a 2025 OT security policy update.",
    },
    {
        "subcategory": "Application crash on uninstall",
        "category": "Software",
        "old_date": "2024-01-15",
        "old_symptoms": "Application crashes during uninstall.",
        "old_resolution": [
            "Manually delete the application folder and related registry keys by hand.",
            "No vendor cleanup tool was available at the time of writing.",
        ],
        "note": "A vendor-provided cleanup/removal tool has since been released and is now the recommended method -- manual registry editing is discouraged under current guidance due to risk of system instability.",
    },
    {
        "subcategory": "Plugin not loading",
        "category": "Software",
        "old_date": "2024-07-01",
        "old_symptoms": "A plugin fails to load in the host application.",
        "old_resolution": [
            "Reinstall the entire host application from scratch.",
            "Plugin-specific reinstallation was not supported at the time.",
        ],
        "note": "Plugin-level reinstall/update is now supported without reinstalling the full host application -- this older guidance recommends an unnecessarily disruptive fix.",
    },
    {
        "subcategory": "Application crash on exit",
        "category": "Software",
        "old_date": "2023-09-14",
        "old_symptoms": "Application crashes when closing.",
        "old_resolution": [
            "No fix available; advise users to save frequently in case of data loss.",
            "Vendor had not yet acknowledged the issue at time of writing.",
        ],
        "note": "The vendor released a patch addressing this in a later update -- current SOP references the fix directly rather than a save-frequently workaround.",
    },
    {
        "subcategory": "Collaboration tool login failure",
        "category": "Software",
        "old_date": "2024-08-20",
        "old_symptoms": "User cannot log into the collaboration tool.",
        "old_resolution": [
            "Advise the user to wait 24 hours, as this was often a temporary platform-side issue historically.",
            "No local troubleshooting steps were documented at the time.",
        ],
        "note": "Current SOP includes specific local cache/token troubleshooting steps that resolve most cases immediately, rather than defaulting to a 24-hour wait.",
    },
    {
        "subcategory": "Application crash on workflow",
        "category": "Software",
        "old_date": "2024-03-30",
        "old_symptoms": "Application crashes during a specific workflow.",
        "old_resolution": [
            "Ask the user to avoid using that workflow feature entirely going forward.",
            "No hotfix was available at the time of writing.",
        ],
        "note": "A hotfix targeting this specific workflow module has since been released -- avoidance is no longer the recommended long-term guidance.",
    },
    {
        "subcategory": "Keyboard malfunction",
        "category": "Hardware",
        "old_date": "2024-05-05",
        "old_symptoms": "Keys on the laptop keyboard are unresponsive.",
        "old_resolution": [
            "Attempt an in-house keyboard replacement using a generic third-party part.",
            "Vendor-certified parts were not required at the time.",
        ],
        "note": "Current policy requires vendor-certified replacement parts only, due to warranty and quality issues experienced with third-party keyboards under this older guidance.",
    },
    {
        "subcategory": "Laptop screen ghost image",
        "category": "Hardware",
        "old_date": "2023-08-19",
        "old_symptoms": "Faint residual image persists on the laptop screen.",
        "old_resolution": [
            "Advise the user this is a known cosmetic issue with no fix; no ticket needed.",
            "Panel replacement was not offered for ghosting alone at the time.",
        ],
        "note": "Panel replacement is now approved for persistent ghosting under current hardware policy -- this older guidance incorrectly treated it as unfixable.",
    },
    {
        "subcategory": "CD/DVD drive not working",
        "category": "Hardware",
        "old_date": "2024-01-08",
        "old_symptoms": "Optical drive fails to read discs.",
        "old_resolution": [
            "Recommend the user purchase a personal external USB optical drive as a permanent workaround.",
            "Internal drive replacement was not stocked as a standard part.",
        ],
        "note": "Internal optical drive replacement is now a stocked standard part -- reimbursing personal purchases is no longer the default guidance.",
    },
    {
        "subcategory": "Laptop screen blue tint",
        "category": "Hardware",
        "old_date": "2024-04-22",
        "old_symptoms": "Laptop screen shows a blue color cast.",
        "old_resolution": [
            "Assume this is always a software calibration issue; do not offer hardware replacement.",
            "No external monitor test step was documented.",
        ],
        "note": "Current SOP includes an external monitor test step to correctly distinguish calibration issues from panel defects -- this older version assumed calibration in all cases, leading to missed hardware replacements.",
    },
    {
        "subcategory": "External HDD not mounting",
        "category": "Hardware",
        "old_date": "2024-07-25",
        "old_symptoms": "External hard drive is not recognized as a mounted drive.",
        "old_resolution": [
            "Advise the user this is usually unfixable and to back up remaining data if possible, then discard the drive.",
            "No Disk Management diagnostic step was documented at the time.",
        ],
        "note": "Current SOP includes a Disk Management check that resolves the majority of these cases without needing to discard the drive -- this older guidance skipped that step entirely.",
    },
    {
        "subcategory": "Proxy server misconfiguration",
        "category": "Network",
        "old_date": "2023-12-11",
        "old_symptoms": "User cannot reach external websites.",
        "old_resolution": [
            "Direct the user to manually edit their browser proxy settings using an outdated internal proxy address.",
            "The current GPO-based proxy push mechanism did not yet exist.",
        ],
        "note": "Proxy configuration is now pushed automatically via GPO -- manual browser edits using the old proxy address referenced here will not work and may cause further connectivity issues.",
    },
    {
        "subcategory": "Switch port failure",
        "category": "Network",
        "old_date": "2024-02-14",
        "old_symptoms": "No connectivity at a specific network port.",
        "old_resolution": [
            "Have the user relocate to a different desk permanently rather than fixing the port.",
            "Port-level diagnostics were handled manually with no ticketing integration.",
        ],
        "note": "Switch port diagnostics are now integrated into the network monitoring dashboard with automated alerts -- relocating the user is no longer the default first response.",
    },
    {
        "subcategory": "Duplicate IP detected",
        "category": "Network",
        "old_date": "2023-11-30",
        "old_symptoms": "Device reports an IP address conflict.",
        "old_resolution": [
            "Assign the device a permanent static IP outside the DHCP range as the standard fix.",
            "DHCP scope review was not part of the standard process at the time.",
        ],
        "note": "Static IP assignment is now discouraged in favor of DHCP reservation -- this older guidance created long-term IP management overhead that current process avoids.",
    },
    {
        "subcategory": "Network IPv6 misconfigured",
        "category": "Network",
        "old_date": "2024-06-01",
        "old_symptoms": "Some services intermittently fail; IPv6 misconfiguration suspected.",
        "old_resolution": [
            "Disable IPv6 org-wide on all devices as the standard fix.",
            "IPv6 was not yet supported on the corporate network at the time.",
        ],
        "note": "The corporate network now supports IPv6 on several segments -- disabling it org-wide is no longer correct and would break services that now depend on it.",
    },
    {
        "subcategory": "Bridge mode error",
        "category": "Network",
        "old_date": "2024-03-19",
        "old_symptoms": "Network adapter reports a bridge-mode-related error.",
        "old_resolution": [
            "Reimage the entire device as the default fix for any adapter configuration error.",
            "Adapter-level bridge configuration was not something desktop support could inspect directly at the time.",
        ],
        "note": "Adapter bridge settings can now be inspected and corrected directly without a full reimage -- this older guidance recommended a far more disruptive fix than necessary.",
    },
]
