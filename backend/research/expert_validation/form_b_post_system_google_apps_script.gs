/*
 * Objective 3 Form B: post-system expert manufacturability review.
 *
 * Use this Google Apps Script only after an expert has completed and frozen
 * Form A. Form B captures supplemental review of the matching system plan for
 * the same specimen; it is not independent expert ground truth.
 *
 * Default safety behavior:
 * - Leave EXISTING_FORM_B_ID blank to create a new final Form B later.
 * - Set EXISTING_FORM_B_ID only if you intentionally want to replace items in
 *   an existing Form B while preserving its response destination.
 * - The normalized response sheet uses the exact CSV schema expected by
 *   backend/research/expert_validation/validate_form_b_responses.py.
 */

const FORM_B_VERSION = "objective3_post_system_form_b_v1";
const FORM_TITLE = "Objective 3 Form B - Post-System Expert Review";
const EXISTING_FORM_B_ID = "";
const EXISTING_RESPONSE_SPREADSHEET_ID = "";
const NORMALIZED_SHEET_NAME = "Form B Normalized Responses";

const FINAL_SPECIMENS = Object.freeze([
  "QZ-01",
  "QZ-03",
  "QZ-05",
  "QZ-08",
  "QZ-14",
]);

const REMOVED_SPECIMENS = Object.freeze([
  "QZ-09",
  "QZ-30",
]);

const SCHEMA_HEADERS = Object.freeze([
  "form_b_response_id",
  "submitted_at",
  "expert_id",
  "expert_name_or_code",
  "form_a_response_id",
  "form_a_completed_before_system_shown",
  "consent_post_system_comparison",
  "form_version",
  "specimen_id",
  "system_plan_reviewed",
  "system_plan_manufacturability",
  "manufacturing_risks_or_practical_concerns",
  "changes_expert_would_make_to_system_plan",
  "revised_retained_weight_estimate_ct",
  "revised_gem_count",
  "revised_recommended_cut_shape",
  "system_orientation_acceptable",
  "orientation_disagreement_explanation",
  "feasibility_confidence_1_to_5",
  "overall_comments",
]);

const IDENTITY_TITLES = Object.freeze({
  expertId: "Expert ID",
  expertName: "Expert name or code",
  formAResponseId: "Original Form A response/submission ID",
  formACompleted: "Form A completion confirmation",
  consent: "Post-system comparison consent",
});

const SPECIMEN_FIELD_TITLES = Object.freeze({
  systemPlanReviewed: "System plan reviewed",
  manufacturability: "System plan manufacturability",
  risks: "Manufacturing risks / practical concerns",
  changes: "Changes you would make to the system plan",
  revisedRetainedWeight: "Revised retained-weight estimate (ct), if changed",
  revisedGemCount: "Revised gem count, if changed",
  revisedCutShape: "Revised recommended cut shape, if changed",
  orientationAcceptable: "System orientation acceptable",
  orientationExplanation: "Orientation disagreement explanation, if any",
  confidence: "Feasibility confidence",
  comments: "Overall comments",
});

function createFinalPostSystemFormB() {
  validateFinalSpecimens_();

  const form = getOrCreateForm_();
  clearFormItems_(form);
  form.setTitle(FORM_TITLE);
  form.setDescription([
    "Form B is a post-system expert review instrument.",
    "Use it only after your Form A recommendation has been completed and frozen.",
    "Review the matching system plan supplied separately for each specimen.",
    "Form B is supplemental manufacturability review and is not an independent baseline.",
    "Final specimens: " + FINAL_SPECIMENS.join(", ") + ".",
  ].join("\n"));
  form.setCollectEmail(false);
  form.setAllowResponseEdits(false);
  form.setLimitOneResponsePerUser(false);
  form.setProgressBar(true);
  form.setShowLinkToRespondAgain(false);
  form.setConfirmationMessage(
    "Thank you. This post-system review will be linked only as supplemental Form B evidence."
  );

  addIdentitySection_(form);
  FINAL_SPECIMENS.forEach(function(specimenId) {
    addSpecimenSection_(form, specimenId);
  });
  validateFormDoesNotContainRemovedSpecimens_(form);

  const spreadsheet = getOrCreateResponseSpreadsheet_();
  form.setDestination(FormApp.DestinationType.SPREADSHEET, spreadsheet.getId());
  ensureNormalizedSheet_(spreadsheet);

  PropertiesService.getScriptProperties().setProperty("FORM_B_ID", form.getId());
  PropertiesService.getScriptProperties().setProperty(
    "FORM_B_RESPONSE_SPREADSHEET_ID",
    spreadsheet.getId()
  );

  Logger.log("Form edit URL: " + form.getEditUrl());
  Logger.log("Form live URL: " + form.getPublishedUrl());
  Logger.log("Response spreadsheet URL: " + spreadsheet.getUrl());
  return {
    formId: form.getId(),
    editUrl: form.getEditUrl(),
    liveUrl: form.getPublishedUrl(),
    spreadsheetId: spreadsheet.getId(),
    spreadsheetUrl: spreadsheet.getUrl(),
  };
}

function installNormalizeFormBTrigger() {
  const formId = getStoredFormId_();
  const form = FormApp.openById(formId);
  ScriptApp.getProjectTriggers().forEach(function(trigger) {
    if (trigger.getHandlerFunction() === "normalizeFormBResponse") {
      ScriptApp.deleteTrigger(trigger);
    }
  });
  ScriptApp.newTrigger("normalizeFormBResponse")
    .forForm(form)
    .onFormSubmit()
    .create();
  Logger.log("Installed normalizeFormBResponse trigger for form " + formId);
}

function normalizeFormBResponse(event) {
  if (!event || !event.response) {
    throw new Error("normalizeFormBResponse requires a Form submit event.");
  }
  const response = event.response;
  const answers = answersByTitle_(response);
  const spreadsheet = SpreadsheetApp.openById(getStoredSpreadsheetId_());
  const sheet = ensureNormalizedSheet_(spreadsheet);

  const formBResponseId = response.getId ? response.getId() : Utilities.getUuid();
  const submittedAt = response.getTimestamp().toISOString();
  const expertId = answer_(answers, IDENTITY_TITLES.expertId);
  const expertName = answer_(answers, IDENTITY_TITLES.expertName);
  const formAResponseId = answer_(answers, IDENTITY_TITLES.formAResponseId);
  const formACompleted = normalizeYesNo_(
    answer_(answers, IDENTITY_TITLES.formACompleted)
  );
  const consent = normalizeYesNo_(answer_(answers, IDENTITY_TITLES.consent));

  const rows = FINAL_SPECIMENS.map(function(specimenId) {
    return [
      formBResponseId,
      submittedAt,
      expertId,
      expertName,
      formAResponseId,
      formACompleted,
      consent,
      FORM_B_VERSION,
      specimenId,
      normalizeYesNo_(answer_(answers, specimenTitle_(
        specimenId,
        SPECIMEN_FIELD_TITLES.systemPlanReviewed
      ))),
      normalizeUpper_(answer_(answers, specimenTitle_(
        specimenId,
        SPECIMEN_FIELD_TITLES.manufacturability
      ))),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.risks)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.changes)),
      answer_(answers, specimenTitle_(
        specimenId,
        SPECIMEN_FIELD_TITLES.revisedRetainedWeight
      )),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.revisedGemCount)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.revisedCutShape)),
      normalizeUpper_(answer_(answers, specimenTitle_(
        specimenId,
        SPECIMEN_FIELD_TITLES.orientationAcceptable
      ))),
      answer_(answers, specimenTitle_(
        specimenId,
        SPECIMEN_FIELD_TITLES.orientationExplanation
      )),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.confidence)),
      answer_(answers, specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.comments)),
    ];
  });

  sheet.getRange(sheet.getLastRow() + 1, 1, rows.length, SCHEMA_HEADERS.length)
    .setValues(rows);
}

function addIdentitySection_(form) {
  form.addSectionHeaderItem()
    .setTitle("Expert identity and Form A linkage")
    .setHelpText("These fields link Form B to the already frozen Form A submission.");
  form.addTextItem()
    .setTitle(IDENTITY_TITLES.expertId)
    .setHelpText("Use the same expert ID/code used for Form A.")
    .setRequired(true);
  form.addTextItem()
    .setTitle(IDENTITY_TITLES.expertName)
    .setHelpText("Name or role-coded identifier.")
    .setRequired(true);
  form.addTextItem()
    .setTitle(IDENTITY_TITLES.formAResponseId)
    .setHelpText("Use the Form A response/submission ID if it was provided.")
    .setRequired(false);
  form.addMultipleChoiceItem()
    .setTitle(IDENTITY_TITLES.formACompleted)
    .setHelpText("Confirm Form A was completed and frozen before any system result was shown.")
    .setChoiceValues(["YES", "NO"])
    .setRequired(true);
  form.addMultipleChoiceItem()
    .setTitle(IDENTITY_TITLES.consent)
    .setHelpText("Consent to link this supplemental post-system review to frozen Form A.")
    .setChoiceValues(["YES", "NO"])
    .setRequired(true);
}

function addSpecimenSection_(form, specimenId) {
  form.addPageBreakItem()
    .setTitle(specimenId)
    .setHelpText("Review the matching system plan supplied separately, then answer for this specimen.");
  form.addMultipleChoiceItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.systemPlanReviewed))
    .setChoiceValues(["YES", "NO"])
    .setRequired(true);
  form.addMultipleChoiceItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.manufacturability))
    .setHelpText("Assess practical manufacturability of the reviewed plan.")
    .setChoiceValues(["YES", "NO", "UNCERTAIN"])
    .setRequired(true);
  form.addParagraphTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.risks))
    .setHelpText("Enter none if no practical concerns are identified.")
    .setRequired(true);
  form.addParagraphTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.changes))
    .setHelpText("Enter no changes if the plan would not be changed.")
    .setRequired(true);
  form.addTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.revisedRetainedWeight))
    .setHelpText("Only if the reviewed plan would be changed.")
    .setRequired(false);
  form.addTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.revisedGemCount))
    .setHelpText("Only if the reviewed plan would be changed.")
    .setRequired(false);
  form.addTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.revisedCutShape))
    .setHelpText("Only if the reviewed plan would be changed.")
    .setRequired(false);
  form.addMultipleChoiceItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.orientationAcceptable))
    .setChoiceValues(["YES", "NO", "UNCERTAIN"])
    .setRequired(true);
  form.addParagraphTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.orientationExplanation))
    .setHelpText("Required when the reviewed orientation is not acceptable.")
    .setRequired(false);
  form.addScaleItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.confidence))
    .setBounds(1, 5)
    .setLabels("Low confidence", "High confidence")
    .setRequired(true);
  form.addParagraphTextItem()
    .setTitle(specimenTitle_(specimenId, SPECIMEN_FIELD_TITLES.comments))
    .setHelpText("Optional overall comments.")
    .setRequired(false);
}

function getOrCreateForm_() {
  if (EXISTING_FORM_B_ID) {
    return FormApp.openById(EXISTING_FORM_B_ID);
  }
  return FormApp.create(FORM_TITLE);
}

function getOrCreateResponseSpreadsheet_() {
  if (EXISTING_RESPONSE_SPREADSHEET_ID) {
    return SpreadsheetApp.openById(EXISTING_RESPONSE_SPREADSHEET_ID);
  }
  return SpreadsheetApp.create("Objective 3 Form B Responses");
}

function clearFormItems_(form) {
  const items = form.getItems();
  for (let index = items.length - 1; index >= 0; index -= 1) {
    form.deleteItem(items[index]);
  }
}

function ensureNormalizedSheet_(spreadsheet) {
  let sheet = spreadsheet.getSheetByName(NORMALIZED_SHEET_NAME);
  if (!sheet) {
    sheet = spreadsheet.insertSheet(NORMALIZED_SHEET_NAME);
  }
  const existingHeader = sheet.getRange(1, 1, 1, SCHEMA_HEADERS.length).getValues()[0];
  const hasHeader = existingHeader.some(function(value) {
    return value !== "";
  });
  if (!hasHeader) {
    sheet.getRange(1, 1, 1, SCHEMA_HEADERS.length).setValues([SCHEMA_HEADERS]);
  }
  const header = sheet.getRange(1, 1, 1, SCHEMA_HEADERS.length).getValues()[0];
  if (header.join("\t") !== SCHEMA_HEADERS.join("\t")) {
    throw new Error(
      "Normalized sheet header does not match backend Form B validation schema."
    );
  }
  return sheet;
}

function answersByTitle_(response) {
  const answers = {};
  response.getItemResponses().forEach(function(itemResponse) {
    answers[itemResponse.getItem().getTitle()] = stringifyAnswer_(
      itemResponse.getResponse()
    );
  });
  return answers;
}

function answer_(answers, title) {
  return answers[title] || "";
}

function stringifyAnswer_(value) {
  if (Array.isArray(value)) {
    return value.join("; ");
  }
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function normalizeUpper_(value) {
  return String(value || "").trim().toUpperCase();
}

function normalizeYesNo_(value) {
  return normalizeUpper_(value) === "YES" ? "YES" : "NO";
}

function specimenTitle_(specimenId, fieldTitle) {
  return specimenId + " - " + fieldTitle;
}

function getStoredFormId_() {
  const formId = EXISTING_FORM_B_ID ||
    PropertiesService.getScriptProperties().getProperty("FORM_B_ID");
  if (!formId) {
    throw new Error("Run createFinalPostSystemFormB first.");
  }
  return formId;
}

function getStoredSpreadsheetId_() {
  const spreadsheetId = EXISTING_RESPONSE_SPREADSHEET_ID ||
    PropertiesService.getScriptProperties().getProperty("FORM_B_RESPONSE_SPREADSHEET_ID");
  if (!spreadsheetId) {
    throw new Error("Run createFinalPostSystemFormB first.");
  }
  return spreadsheetId;
}

function validateFinalSpecimens_() {
  REMOVED_SPECIMENS.forEach(function(specimenId) {
    if (FINAL_SPECIMENS.indexOf(specimenId) !== -1) {
      throw new Error("Removed specimen appears in final Form B set: " + specimenId);
    }
  });
  if (FINAL_SPECIMENS.length !== 5) {
    throw new Error("Final Form B specimen set must contain exactly five specimens.");
  }
}

function validateFormDoesNotContainRemovedSpecimens_(form) {
  const text = form.getItems().map(function(item) {
    return item.getTitle() + " " + item.getHelpText();
  }).join("\n");
  REMOVED_SPECIMENS.forEach(function(specimenId) {
    if (text.indexOf(specimenId) !== -1) {
      throw new Error("Form still contains removed specimen: " + specimenId);
    }
  });
}
