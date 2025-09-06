import { CompanyReportData } from "../types";

type JSONResponse = {
  output?: string[];
  errors?: Array<{ message: string }>;
};

export const getCompanies = async () => {
  const response = await fetch(
    `${import.meta.env.VITE_REPORT_DATA_ENDPOINT}/companyReport/list`,
    {
      method: "POST",
    }
  );

  if (!response.ok) {
    return Promise.reject(
      new Error(`Failed to get company: ${response.statusText}`)
    );
  }
  const { output, errors }: JSONResponse = await response.json();

  if (errors !== undefined) {
    const error = new Error(
      errors?.map((e) => e.message).join("\n") ?? "unknown"
    );
    return Promise.reject(error);
  }
  console.log("data", output);
  return output ?? undefined;
};

// Define the type for the JSON data
type Chemical = {
  name: string;
  class?: string;
  weight?: string;
  example?: string;
  examples?: string[];
};

type FunctionalRole = {
  role: string;
  total_weight: string;
  chemicals: Chemical[];
};

export type CosmeticProduct = {
  entity: string;
  type: string;
  properties: {
    product_name: string;
    functional_roles: FunctionalRole[];
  };
};

export const get_patent_data = (): CosmeticProduct => {
  // Hardcoded JSON object
  const data: CosmeticProduct = {
    entity: "cosmetic_product",
    type: "product",
    properties: {
      product_name: "cosmetic nail composition",
      functional_roles: [
        {
          role: "UV Gel Portion",
          total_weight: "19-50% by weight",
          chemicals: [
            {
              name: "Difunctional Acrylate Oligomer",
              class: "Urethane Monomer",
              weight: "8-15% by weight",
            },
            {
              name: "Polymerizable Methacrylate",
              class: "Pyromellitic Dimethacrylate",
              weight: "2-8% by weight",
            },
            {
              name: "Photoinitiator",
              class: "Type I",
              example: "Ethyl Trimethylbenzoyl Phenylphosphinate",
              weight: "3-6% by weight",
            },
          ],
        },
        {
          role: "Solvent-Based Portion",
          total_weight: "50-70% by weight",
          chemicals: [
            {
              name: "Acetates",
              class: "C1-C4 Alkyl Acetates",
              weight: "45-60% by weight",
            },
            {
              name: "C2-C5 Monoalcohols",
              examples: ["Ethanol", "Isopropanol"],
              weight: "3-5% by weight",
            },
            {
              name: "Volatile Hydrocarbon",
              example: "Isododecane",
              weight: "1-2% by weight",
            },
            {
              name: "Benzoic/Citric/Isobutyrate Derivatives",
              examples: [
                "Dipropylene Glycol Dibenzoate",
                "Acetyl Tributyl Citrate",
              ],
              weight: "3-5% by weight",
            },
          ],
        },
        {
          role: "Film Forming Agents",
          total_weight: "5-20% by weight",
          chemicals: [
            {
              name: "Celluloses",
              examples: ["Nitrocellulose", "Cellulose Acetate Butyrate"],
              weight: "3-10% by weight",
            },
            {
              name: "Acrylates",
              examples: [
                "Acrylates Copolymer",
                "Acrylates/Isobornyl Acrylate Copolymer",
              ],
              weight: "1-2% by weight",
            },
            {
              name: "Co-Film Forming Agents",
              examples: [
                "Tosylamide Epoxy Resin",
                "Adipic Acid/Neopentyl Glycol Copolymer",
              ],
              weight: "3-10% by weight",
            },
          ],
        },
        {
          role: "Colorants",
          total_weight: "0.5-2% by weight",
          chemicals: [
            {
              name: "Titanium Dioxide",
              class: "Pigment",
            },
            {
              name: "D&C Red 7",
              class: "Pigment",
            },
            {
              name: "Bismuth Oxychloride",
              class: "Pigment",
            },
          ],
        },
        {
          role: "Additional Components",
          total_weight: "<5% by weight",
          chemicals: [
            {
              name: "UV Absorbing Agents",
              examples: [
                "Butyl Methoxydibenzoylmethane",
                "Ethylhexyl Methoxycinnamate",
              ],
              weight: "<1% by weight",
            },
            {
              name: "Fillers",
              examples: ["Talc", "Silica", "Nylon-12"],
              weight: "<2% by weight",
            },
            {
              name: "pH Adjusting Agents",
              examples: ["Citric Acid", "Sodium Metasilicate"],
              weight: "<1% by weight",
            },
          ],
        },
      ],
    },
  };

  // Return the hardcoded data
  return data;
};
