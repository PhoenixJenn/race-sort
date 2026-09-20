"""Event-scoped, multi-variant identity registry for RaceSort.

This model-free application module represents one race-number string as one or
more visually distinct vehicle/rider variants. Each variant can retain several
confirmed crop references and metadata without moving or modifying the source
photographs. Start with ``EventRegistry`` when reading this file.
"""

from dataclasses import asdict, dataclass, field
from datetime import date
import json
from pathlib import Path

from racesort.identifiers import normalize_number


NUMBER_ACTIONS = {"ACCEPT", "CORRECT", "REJECT"}
VARIANT_ACTIONS = {"MATCH_EXISTING", "CREATE_NEW"}
SUPPORTED_RACE_TYPES = {"motorcycle", "car"}
SUPPORTED_GROUPS = {"A", "B", "C"}


def require_text(value, field_name):
    """Return stripped text or raise a field-specific validation error."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def require_race_number(value):
    """Validate a race-number string without accepting numeric Python values."""

    if not isinstance(value, str):
        raise ValueError("race_number must be a string")
    normalized = normalize_number(value)
    if normalized is None:
        raise ValueError("race_number is invalid")
    return normalized


def validate_session_context(group, cycle):
    """Validate optional group/cycle context without making it identity."""

    if group is not None and group not in SUPPORTED_GROUPS:
        raise ValueError("group must be A, B, C, or None")
    if cycle is not None and cycle not in range(1, 6):
        raise ValueError("cycle must be from 1 through 5 or None")


@dataclass(frozen=True)
class ConfirmedReference:
    """One confirmed vehicle crop and its preserved provenance."""

    source_photo: str
    crop: str
    confirmation_source: str
    group: str | None = None
    cycle: int | None = None
    session_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "source_photo",
            require_text(self.source_photo, "source_photo"),
        )
        object.__setattr__(self, "crop", require_text(self.crop, "crop"))
        object.__setattr__(
            self,
            "confirmation_source",
            require_text(self.confirmation_source, "confirmation_source"),
        )
        validate_session_context(self.group, self.cycle)

    @property
    def reference_key(self):
        """Identify the crop without interpreting or modifying its path."""

        return self.source_photo, self.crop


@dataclass
class VehicleVariant:
    """One visually distinct motorcycle/rider or car using a race number."""

    variant_id: str
    vehicle_type: str
    metadata: dict = field(default_factory=dict)
    references: list[ConfirmedReference] = field(default_factory=list)

    def __post_init__(self):
        self.variant_id = require_text(self.variant_id, "variant_id")
        self.vehicle_type = require_text(self.vehicle_type, "vehicle_type")


@dataclass
class RegistryEntry:
    """All known visual variants that share one race-number string."""

    race_number: str
    variants: dict[str, VehicleVariant] = field(default_factory=dict)

    def __post_init__(self):
        self.race_number = require_race_number(self.race_number)


@dataclass
class EventRegistry:
    """Registry of confirmed identities for one event."""

    event_id: str
    race_type: str
    event_date: str | None = None
    entries: dict[str, RegistryEntry] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self):
        self.event_id = require_text(self.event_id, "event_id")
        self.race_type = require_text(self.race_type, "race_type")
        if self.race_type not in SUPPORTED_RACE_TYPES:
            raise ValueError("race_type must be motorcycle or car")
        if self.event_date is not None:
            try:
                date.fromisoformat(self.event_date)
            except (TypeError, ValueError) as exc:
                raise ValueError("event_date must use YYYY-MM-DD") from exc

    def add_variant(self, race_number, variant_id, metadata=None):
        """Add a distinct variant under a race number and return it."""

        race_number = require_race_number(race_number)
        variant_id = require_text(variant_id, "variant_id")
        entry = self.entries.setdefault(race_number, RegistryEntry(race_number))
        if variant_id in entry.variants:
            raise ValueError(
                f"variant {variant_id!r} already exists for {race_number!r}"
            )

        variant = VehicleVariant(
            variant_id=variant_id,
            vehicle_type=self.race_type,
            metadata=dict(metadata or {}),
        )
        entry.variants[variant_id] = variant
        return variant

    def add_reference(self, race_number, variant_id, reference):
        """Attach one confirmed reference without duplicating/reassigning it."""

        race_number = require_race_number(race_number)
        entry = self.entries.get(race_number)
        if entry is None or variant_id not in entry.variants:
            raise KeyError(f"unknown variant {race_number!r}/{variant_id!r}")

        for existing_number, existing_entry in self.entries.items():
            for existing_variant in existing_entry.variants.values():
                for existing_reference in existing_variant.references:
                    if existing_reference.reference_key == reference.reference_key:
                        if (
                            existing_number == race_number
                            and existing_variant.variant_id == variant_id
                        ):
                            return False
                        raise ValueError(
                            "a crop cannot belong to multiple registry variants"
                        )

        entry.variants[variant_id].references.append(reference)
        return True

    def to_dict(self):
        """Return a JSON-ready representation with explicit number entries."""

        return {
            "schema_version": self.schema_version,
            "event": {
                "event_id": self.event_id,
                "event_date": self.event_date,
                "race_type": self.race_type,
            },
            "numbers": {
                number: {
                    "race_number": entry.race_number,
                    "variants": {
                        variant_id: asdict(variant)
                        for variant_id, variant in entry.variants.items()
                    },
                }
                for number, entry in self.entries.items()
            },
        }

    @classmethod
    def from_dict(cls, data):
        """Validate and reconstruct a registry from decoded JSON data."""

        if not isinstance(data, dict):
            raise ValueError("registry data must be an object")
        if data.get("schema_version") != 1:
            raise ValueError("unsupported registry schema_version")

        event = data.get("event")
        numbers = data.get("numbers")
        if not isinstance(event, dict) or not isinstance(numbers, dict):
            raise ValueError("registry event and numbers must be objects")

        try:
            registry = cls(
                event_id=event["event_id"],
                event_date=event.get("event_date"),
                race_type=event["race_type"],
                schema_version=1,
            )
        except KeyError as exc:
            raise ValueError(f"missing registry event field: {exc.args[0]}") from exc

        for number_key, entry_data in numbers.items():
            if not isinstance(entry_data, dict):
                raise ValueError("each number entry must be an object")
            race_number = require_race_number(entry_data.get("race_number"))
            if number_key != race_number:
                raise ValueError("number key must match canonical race_number")

            variants = entry_data.get("variants")
            if not isinstance(variants, dict):
                raise ValueError("entry variants must be an object")

            for variant_key, variant_data in variants.items():
                if not isinstance(variant_data, dict):
                    raise ValueError("each variant must be an object")
                if variant_data.get("variant_id") != variant_key:
                    raise ValueError("variant key must match variant_id")
                if variant_data.get("vehicle_type") != registry.race_type:
                    raise ValueError("variant vehicle_type must match event race_type")

                metadata = variant_data.get("metadata", {})
                references = variant_data.get("references", [])
                if not isinstance(metadata, dict):
                    raise ValueError("variant metadata must be an object")
                if not isinstance(references, list):
                    raise ValueError("variant references must be a list")

                registry.add_variant(race_number, variant_key, metadata)
                for reference_data in references:
                    if not isinstance(reference_data, dict):
                        raise ValueError("each confirmed reference must be an object")
                    reference_metadata = reference_data.get("metadata", {})
                    if not isinstance(reference_metadata, dict):
                        raise ValueError("reference metadata must be an object")
                    try:
                        reference = ConfirmedReference(
                            source_photo=reference_data["source_photo"],
                            crop=reference_data["crop"],
                            confirmation_source=reference_data[
                                "confirmation_source"
                            ],
                            group=reference_data.get("group"),
                            cycle=reference_data.get("cycle"),
                            session_id=reference_data.get("session_id"),
                            metadata=reference_metadata,
                        )
                    except KeyError as exc:
                        raise ValueError(
                            f"missing confirmed reference field: {exc.args[0]}"
                        ) from exc
                    registry.add_reference(race_number, variant_key, reference)

        return registry

    @classmethod
    def load(cls, path):
        """Load and validate a registry JSON file."""

        try:
            with Path(path).open(encoding="utf-8") as registry_file:
                data = json.load(registry_file)
        except json.JSONDecodeError as exc:
            raise ValueError("registry file is not valid JSON") from exc
        return cls.from_dict(data)

    def save(self, path):
        """Atomically save generated registry JSON without touching photos."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")

        try:
            with temporary_path.open("w", encoding="utf-8") as registry_file:
                json.dump(self.to_dict(), registry_file, indent=2)
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


@dataclass(frozen=True)
class HumanConfirmation:
    """One human number decision plus its variant assignment intent."""

    source_photo: str
    crop: str
    number_action: str
    proposed_number: str | None = None
    corrected_number: str | None = None
    variant_action: str | None = None
    variant_id: str | None = None
    group: str | None = None
    cycle: int | None = None
    session_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "source_photo",
            require_text(self.source_photo, "source_photo"),
        )
        object.__setattr__(self, "crop", require_text(self.crop, "crop"))
        if self.number_action not in NUMBER_ACTIONS:
            raise ValueError("number_action must be ACCEPT, CORRECT, or REJECT")
        validate_session_context(self.group, self.cycle)

        if self.number_action == "REJECT":
            if self.variant_action is not None or self.variant_id is not None:
                raise ValueError("rejected confirmations cannot assign a variant")
            return

        if self.variant_action not in VARIANT_ACTIONS:
            raise ValueError(
                "variant_action must be MATCH_EXISTING or CREATE_NEW"
            )
        object.__setattr__(
            self,
            "variant_id",
            require_text(self.variant_id, "variant_id"),
        )
        self.resolved_number()

    def resolved_number(self):
        """Return the confirmed string number, or None for rejection."""

        if self.number_action == "REJECT":
            return None
        if self.number_action == "ACCEPT":
            return require_race_number(self.proposed_number)
        return require_race_number(self.corrected_number)


def apply_human_confirmation(
    registry,
    confirmation,
    *,
    variant_metadata=None,
):
    """Apply one human confirmation to the registry and return its identity."""

    race_number = confirmation.resolved_number()
    if race_number is None:
        return None

    if confirmation.variant_action == "CREATE_NEW":
        registry.add_variant(
            race_number,
            confirmation.variant_id,
            metadata=variant_metadata,
        )

    confirmed_reference = ConfirmedReference(
        source_photo=confirmation.source_photo,
        crop=confirmation.crop,
        confirmation_source="human",
        group=confirmation.group,
        cycle=confirmation.cycle,
        session_id=confirmation.session_id,
        metadata=dict(confirmation.metadata),
    )
    registry.add_reference(
        race_number,
        confirmation.variant_id,
        confirmed_reference,
    )
    return race_number, confirmation.variant_id
